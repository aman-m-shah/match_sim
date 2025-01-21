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
    
    # Base score components
    score_components = {
        'interview': applicant.interview_score * weights.interview_weight,
        'letters': applicant.letter_scores * weights.letters_weight,
    }
    
    # Add exam scores with configurable weights
    if 'step2_ck' in normalized_scores:
        score_components['step2'] = normalized_scores['step2_ck'] * weights.step2_weight
    if 'comlex2' in normalized_scores:
        score_components['comlex2'] = normalized_scores['comlex2'] * weights.comlex2_weight
    
    # Research impact with configurable weight
    research_score = min(1.0, applicant.publications / 5)  # Cap at 5 publications
    score_components['research'] = research_score * weights.research_weight
    
    # Program fit bonuses with configurable weights
    if program.research_focus and applicant.research_focus:
        score_components['research_alignment'] = weights.research_alignment_bonus
    if applicant.away_rotation:
        score_components['away_rotation'] = weights.away_rotation_bonus
    if applicant.alumni_connection:
        score_components['alumni'] = weights.alumni_bonus
    if program.geographic_preference:
        score_components['geographic'] = weights.geographic_bonus
    
    # Calculate base final score
    base_final_score = sum(score_components.values())
    
    # Apply configurable program tier adjustment
    tier_factors = {
        1: weights.tier_1_factor,
        2: weights.tier_2_factor,
        3: weights.tier_3_factor
    }
    base_final_score *= tier_factors[program.program_tier]
    
    # Add configurable random variance
    random_variance = random.uniform(
        -weights.score_random_factor,
        weights.score_random_factor
    ) * base_final_score
    final_score = base_final_score + random_variance
    
    # Ensure final score remains between 0 and 1
    return min(1.0, max(0.0, final_score))

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
    
    # Sort by score with configurable random factor
    ranked_applicants = []
    for app_id, score in scores:
        if score > 0:  # Skip zero scores
            random_adj = random.uniform(
                -weights.rank_random_factor,
                weights.rank_random_factor
            ) * score
            ranked_applicants.append((app_id, score + random_adj))
    
    return [
        app_id for app_id, _ in
        sorted(ranked_applicants, key=lambda x: x[1], reverse=True)
    ]

def run_matching_round(
    applicant_preferences: Dict[str, List[str]],
    program_preferences: Dict[str, List[str]],
    program_capacities: Dict[str, int],
    weights: ScoreWeights
) -> MatchResult:
    """Run matching algorithm with configurable randomness."""
    matches = {prog: [] for prog in program_capacities.keys()}
    unmatched = set(applicant_preferences.keys())
    next_choices = {app: 0 for app in applicant_preferences.keys()}
    
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
            
            rankings = {
                app: ranked_applicants.index(app)
                if app in ranked_applicants else float('inf')
                for app in current_matches + [applicant]
            }
            
            # Add configurable random variation to rankings
            sorted_applicants = sorted(
                rankings.keys(),
                key=lambda x: (
                    rankings[x] +
                    random.uniform(
                        -weights.match_random_factor,
                        weights.match_random_factor
                    ) * len(ranked_applicants)
                )
            )
            
            capacity = program_capacities[program]
            accepted = sorted_applicants[:capacity]
            
            if applicant in accepted:
                rejected = sorted_applicants[capacity]
                matches[program] = accepted
                unmatched.add(rejected)
            else:
                unmatched.add(applicant)
    
    # Create match result
    applicant_matches = {}
    for prog, matched_apps in matches.items():
        for app in matched_apps:
            applicant_matches[app] = prog
            
    unfilled = {
        prog: capacity - len(matches[prog])
        for prog, capacity in program_capacities.items()
    }
    
    return MatchResult(
        applicant_matches=applicant_matches,
        program_matches=matches,
        unmatched_applicants=list(set(applicant_preferences.keys()) - set(applicant_matches.keys())),
        unfilled_positions=unfilled
    )
