from typing import Dict, List, Tuple
import random
from models import Applicant, Program, Specialty, MatchResult, ScoreWeights

def calculate_applicant_score(
    applicant: Applicant,
    program: Program,
    specialty: Specialty,
    weights: ScoreWeights
) -> float:
    """Calculate program's preference score for an applicant using configurable weights."""
    # Check DO status compatibility
    if applicant.is_do and not program.is_do_friendly:
        return 0.0
    
    # Get normalized exam scores
    normalized_scores = applicant.exam_scores.normalize_scores()
    
    # Initialize base score
    score = 0.0
    max_score = 0.0  # Track maximum possible score
    
    # Add exam scores - take the better of STEP2 or COMLEX2 if both exist
    exam_score = 0.0
    if 'step2_ck' in normalized_scores and 'comlex2' in normalized_scores:
        # If both exams exist, use the better score
        exam_score = max(
            normalized_scores['step2_ck'],
            normalized_scores['comlex2']
        )
        score += exam_score * (weights.step2_weight + weights.comlex2_weight)
        max_score += weights.step2_weight + weights.comlex2_weight
    elif 'step2_ck' in normalized_scores:
        exam_score = normalized_scores['step2_ck']
        score += exam_score * weights.step2_weight
        max_score += weights.step2_weight
    elif 'comlex2' in normalized_scores:
        exam_score = normalized_scores['comlex2']
        score += exam_score * weights.comlex2_weight
        max_score += weights.comlex2_weight
    
    # Add interview score (highly weighted for strong performers)
    interview_impact = applicant.interview_score * weights.interview_weight
    if applicant.interview_score > 0.8:  # Bonus for exceptional interviews
        interview_impact *= 1.2
    score += interview_impact
    max_score += weights.interview_weight
    
    # Add letter scores
    letter_impact = applicant.letter_scores * weights.letters_weight
    if applicant.letter_scores > 0.8:  # Bonus for exceptional letters
        letter_impact *= 1.2
    score += letter_impact
    max_score += weights.letters_weight
    
    # Research impact
    research_score = min(1.0, applicant.publications / 5)  # Cap at 5 publications
    score += research_score * weights.research_weight
    max_score += weights.research_weight
    
    # Normalize base score
    if max_score > 0:
        score = score / max_score
    
    # Calculate and apply bonuses
    bonus_score = 0.0
    
    # Research alignment bonus (increased for research-focused programs)
    if program.research_focus and applicant.research_focus:
        research_bonus = weights.research_alignment_bonus
        if research_score > 0.6:  # Extra bonus for significant research
            research_bonus *= 1.5
        bonus_score += research_bonus
    
    # Away rotation bonus (slightly higher impact)
    if applicant.away_rotation:
        bonus_score += weights.away_rotation_bonus * 1.2
    
    # Alumni connection bonus
    if applicant.alumni_connection:
        bonus_score += weights.alumni_bonus
    
    # Geographic preference bonus
    if program.geographic_preference:
        bonus_score += weights.geographic_bonus
    
    # Cap total bonus at 40% of base score
    max_bonus = score * 0.4
    bonus_score = min(max_bonus, bonus_score)
    
    # Add capped bonus to score
    score = min(1.0, score + bonus_score)
    
    # Apply program tier adjustment (with slightly reduced impact)
    tier_factors = {
        1: weights.tier_1_factor,
        2: weights.tier_2_factor * 1.1,  # Boost tier 2 slightly
        3: weights.tier_3_factor * 1.2   # Boost tier 3 slightly
    }
    score *= tier_factors[program.program_tier]
    
    # Add smaller random variance for high scores
    if score > 0:
        variance_factor = weights.score_random_factor
        if score > 0.8:  # Reduce randomness for high scores
            variance_factor *= 0.5
        random_variance = random.uniform(-variance_factor, variance_factor) * score
        score = score + random_variance
    
    # Ensure final score remains between 0 and 1
    return min(1.0, max(0.0, score))

def generate_rank_order(
    applicants: Dict[str, Applicant],
    program: Program,
    specialty: Specialty,
    weights: ScoreWeights
) -> List[str]:
    """Generate program's rank order list with configurable weights and randomness."""
    scores = []
    for app_id, applicant in applicants.items():
        score = calculate_applicant_score(
            applicant,
            program,
            specialty,
            weights
        )
        scores.append((app_id, score))
    
    # Sort by score with reduced randomness for high scores
    ranked_applicants = []
    for app_id, score in scores:
        if score > 0:  # Skip zero scores
            variance_factor = weights.rank_random_factor
            if score > 0.8:  # Reduce randomness for high scores
                variance_factor *= 0.5
            random_adj = random.uniform(-variance_factor, variance_factor) * score
            ranked_applicants.append((app_id, score + random_adj))
    
    # Sort by final score
    return [
        app_id for app_id, _ in 
        sorted(ranked_applicants, key=lambda x: x[1], reverse=True)
    ]

def run_matching_round(
    applicant_preferences: Dict[str, List[str]],
    program_preferences: Dict[str, Dict],
    program_capacities: Dict[str, int],
    weights: ScoreWeights
) -> MatchResult:
    """Run matching algorithm with configurable randomness."""
    # Initialize matches dictionary and other trackers
    matches = {prog: [] for prog in program_capacities.keys()}
    unmatched = set(applicant_preferences.keys())
    next_choices = {app: 0 for app in applicant_preferences.keys()}
    
    # Run matching algorithm
    while unmatched:
        applicant = unmatched.pop()
        rank_list = applicant_preferences[applicant]
        
        if next_choices[applicant] >= len(rank_list):
            continue
            
        program = rank_list[next_choices[applicant]]
        next_choices[applicant] += 1
        
        if len(matches[program]) < program_capacities[program]:
            matches[program].append(applicant)
        else:
            current_matches = matches[program]
            ranked_applicants = program_preferences[program]
            
            # Calculate rankings with reduced randomness for higher ranks
            rankings = {}
            for app in current_matches + [applicant]:
                if app in ranked_applicants:
                    rank_pos = ranked_applicants.index(app)
                    random_factor = weights.match_random_factor
                    if rank_pos < len(ranked_applicants) // 4:  # Top 25% of ranks
                        random_factor *= 0.5
                    rankings[app] = rank_pos + random.uniform(
                        -random_factor, random_factor
                    ) * min(5, rank_pos + 1)
                else:
                    rankings[app] = float('inf')
            
            # Sort by adjusted rankings
            sorted_applicants = sorted(rankings.keys(), key=lambda x: rankings[x])
            
            # Accept top candidates up to capacity
            capacity = program_capacities[program]
            accepted = sorted_applicants[:capacity]
            
            if applicant in accepted:
                rejected = sorted_applicants[capacity]
                matches[program] = accepted
                unmatched.add(rejected)
            else:
                unmatched.add(applicant)
    
    # Create applicant->program matches dictionary
    applicant_matches = {}
    for prog, matched_apps in matches.items():
        for app in matched_apps:
            applicant_matches[app] = prog
    
    # Calculate unfilled positions
    unfilled = {
        prog: capacity - len(matches[prog])
        for prog, capacity in program_capacities.items()
    }
    
    # Return match result
    return MatchResult(
        applicant_matches=applicant_matches,
        program_matches=matches,
        unmatched_applicants=list(set(applicant_preferences.keys()) - set(applicant_matches.keys())),
        unfilled_positions=unfilled
    )
