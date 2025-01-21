import numpy as np
from typing import Dict, List, Tuple
from models import Applicant, Program, Specialty, ExamScores
from matcher import calculate_applicant_score, generate_rank_order, run_matching_round

def generate_other_applicants(
    programs: Dict[str, Program],
    specialties: Dict[str, Specialty],
    n_per_specialty: int = None
) -> Dict[str, Applicant]:
    """Generate simulated competing applicants."""
    other_applicants = {}
    averages = ExamScores.get_national_averages()
    
    for specialty_name, specialty in specialties.items():
        if n_per_specialty is None:
            n_applicants = max(50, specialty.total_applicants - 1)  # -1 for our applicant
        else:
            n_applicants = n_per_specialty
        
        for i in range(n_applicants):
            # Determine if DO student
            is_do = np.random.random() < 0.20  # 20% DO students
            
            # Generate exam scores based on specialty competitiveness
            competitiveness = specialty.competitiveness_factor
            step2_mean = averages['step2_ck']['mean'] + (competitiveness * 10)
            step2_std = averages['step2_ck']['std'] * 0.8
            
            if is_do:
                # DO students might or might not have Step 2
                has_step2 = np.random.random() < 0.7  # 70% have Step 2
                step2_score = int(np.random.normal(step2_mean-5, step2_std)) if has_step2 else None
                
                # All DOs have COMLEX
                comlex_mean = averages['comlex2']['mean'] + (competitiveness * 50)
                comlex_std = averages['comlex2']['std'] * 0.8
                comlex2_score = int(np.random.normal(comlex_mean, comlex_std))
            else:
                # MD students have Step 2
                step2_score = int(np.random.normal(step2_mean, step2_std))
                comlex2_score = None
            
            # Research more likely in competitive/research-focused specialties
            research_prob = (specialty.competitiveness_factor + specialty.research_emphasis) / 2
            n_publications = np.random.poisson(research_prob * 3)
            
            # Generate applicant
            applicant = Applicant(
                name=f"applicant_{specialty_name}_{i}",
                rank_list=[],  # Will be filled later
                interview_score=np.random.beta(2, 2),  # Beta distribution for scores
                exam_scores=ExamScores(
                    step2_ck=step2_score,
                    comlex2=comlex2_score
                ),
                letter_scores=np.random.beta(2, 2),
                publications=n_publications,
                is_do=is_do,
                research_focus=(n_publications >= 2),
                away_rotation=np.random.random() < 0.2,
                alumni_connection=np.random.random() < 0.1
            )
            
            other_applicants[f"applicant_{specialty_name}_{i}"] = applicant
    
    return other_applicants

def generate_applicant_preferences(
    applicants: Dict[str, Applicant],
    programs: Dict[str, Program],
    specialties: Dict[str, Specialty]
) -> Dict[str, List[str]]:
    """Generate preference lists for simulated applicants."""
    preferences = {}
    
    for app_id, applicant in applicants.items():
        if app_id == 'our_applicant':
            # Use actual rank list for our applicant
            preferences[app_id] = applicant.rank_list
            continue
        
        # For other applicants, generate reasonable preferences
        scores = []
        for prog_name, program in programs.items():
            specialty = specialties[program.specialty]
            
            # Skip if DO applicant and non-DO-friendly program
            if applicant.is_do and not program.is_do_friendly:
                continue
                
            # Calculate base desirability
            score = 0.0
            
            # Program tier is important (1 is best, 3 is worst)
            score += (4 - program.program_tier) * 0.3  # Fixed: program_tier instead of tier
            
            # Research alignment matters
            if applicant.research_focus and program.research_focus:
                score += 0.2
                
            # Away rotation and alumni connections matter
            if applicant.away_rotation:
                score += 0.15
            if applicant.alumni_connection:
                score += 0.1
                
            # Add some randomness
            score += np.random.normal(0, 0.1)
            
            scores.append((prog_name, score))
        
        # Sort by score and take top programs
        n_ranks = np.random.randint(5, 15)  # Random number of ranks
        preferences[app_id] = [
            prog for prog, _ in 
            sorted(scores, key=lambda x: x[1], reverse=True)
        ][:n_ranks]
    
    return preferences

def simulate_match(
    our_applicant: Applicant,
    programs: Dict[str, Program],
    specialties: Dict[str, Specialty],
    n_simulations: int = 1000,
    progress_callback = None
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Run multiple match simulations to calculate probabilities."""
    match_counts = {prog: 0 for prog in programs}
    match_counts['No Match'] = 0
    
    detailed_stats = {
        prog: {
            'scores': [],
            'ranks': []
        } for prog in programs
    }
    
    for i in range(n_simulations):
        if progress_callback:
            progress_callback(i + 1, n_simulations)
        
        # Generate other applicants
        other_applicants = generate_other_applicants(programs, specialties)
        all_applicants = {'our_applicant': our_applicant, **other_applicants}
        
        # Generate preferences
        applicant_prefs = generate_applicant_preferences(
            all_applicants, programs, specialties
        )
        
        # Generate program preferences
        program_prefs = {
            prog_name: generate_rank_order(all_applicants, program, specialties[program.specialty])
            for prog_name, program in programs.items()
        }
        
        # Run the match
        result = run_matching_round(
            applicant_prefs,
            program_prefs,
            {p.name: p.spots_available for p in programs.values()}
        )
        
        # Record results
        matched_program = result.applicant_matches.get('our_applicant', 'No Match')
        match_counts[matched_program] += 1
        
        if matched_program != 'No Match':
            program = programs[matched_program]
            specialty = specialties[program.specialty]
            
            # Record score and rank
            score = calculate_applicant_score(our_applicant, program, specialty)
            rank = our_applicant.rank_list.index(matched_program) + 1
            
            detailed_stats[matched_program]['scores'].append(score)
            detailed_stats[matched_program]['ranks'].append(rank)
    
    # Calculate probabilities and statistics
    probabilities = {
        name: count/n_simulations 
        for name, count in match_counts.items()
    }
    
    # Calculate averages for detailed stats
    for prog in detailed_stats:
        scores = detailed_stats[prog]['scores']
        ranks = detailed_stats[prog]['ranks']
        if scores:
            detailed_stats[prog]['avg_score'] = np.mean(scores)
            detailed_stats[prog]['avg_rank'] = np.mean(ranks)
            detailed_stats[prog]['std_score'] = np.std(scores)
            detailed_stats[prog]['std_rank'] = np.std(ranks)
        else:
            detailed_stats[prog]['avg_score'] = 0
            detailed_stats[prog]['avg_rank'] = 0
            detailed_stats[prog]['std_score'] = 0
            detailed_stats[prog]['std_rank'] = 0
    
    return probabilities, detailed_stats