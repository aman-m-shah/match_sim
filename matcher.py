from typing import Dict, List, Tuple
from models import Applicant, Program, Specialty, MatchResult

def calculate_applicant_score(applicant: Applicant, program: Program, specialty: Specialty) -> float:
    """Calculate program's preference score for an applicant."""
    # Get normalized exam scores
    normalized_scores = applicant.exam_scores.normalize_scores()
    
    # Check DO status compatibility
    if applicant.is_do and not program.is_do_friendly:
        return 0.0
    
    # Base score components
    score_components = {
        'interview': applicant.interview_score * 0.30,  # Interview is key
        'letters': applicant.letter_scores * 0.15,
    }
    
    # Add exam scores
    if 'step2_ck' in normalized_scores:
        score_components['step2'] = normalized_scores['step2_ck'] * 0.25
    if 'comlex2' in normalized_scores:
        score_components['comlex2'] = normalized_scores['comlex2'] * 0.25
    
    # Research impact - more important for research focused programs
    research_weight = 0.20 if program.research_focus else 0.10
    research_score = min(1.0, applicant.publications / 5)  # Cap at 5 publications
    score_components['research'] = research_score * research_weight
    
    # Program fit bonuses
    if program.research_focus and applicant.research_focus:
        score_components['research_alignment'] = 0.10
    if applicant.away_rotation:
        score_components['away_rotation'] = 0.15
    if applicant.alumni_connection:
        score_components['alumni'] = 0.05
        
    # Calculate final score
    final_score = sum(score_components.values())
    
    # Apply program tier adjustment
    tier_factor = {1: 1.0, 2: 0.9, 3: 0.8}[program.program_tier]
    final_score *= tier_factor
    
    return min(1.0, final_score)

def generate_rank_order(
    applicants: Dict[str, Applicant],
    program: Program,
    specialty: Specialty
) -> List[str]:
    """Generate a program's rank order list of applicants."""
    # Calculate scores for all applicants
    scores = []
    for app_id, applicant in applicants.items():
        score = calculate_applicant_score(applicant, program, specialty)
        scores.append((app_id, score))
    
    # Sort by score, highest to lowest, and filter out zero scores
    ranked_applicants = [
        app_id for app_id, score in 
        sorted(scores, key=lambda x: x[1], reverse=True)
        if score > 0
    ]
    
    return ranked_applicants

def run_matching_round(
    applicant_preferences: Dict[str, List[str]],
    program_preferences: Dict[str, Dict[str, List[str]]],
    program_capacities: Dict[str, int]
) -> MatchResult:
    """Run one round of the matching algorithm."""
    # Initialize matches
    matches = {prog: [] for prog in program_capacities.keys()}
    unmatched = set(applicant_preferences.keys())
    
    # Track how far each applicant has gone down their rank list
    next_choices = {app: 0 for app in applicant_preferences.keys()}
    
    while unmatched:
        applicant = unmatched.pop()
        rank_list = applicant_preferences[applicant]
        
        # If applicant has exhausted their rank list
        if next_choices[applicant] >= len(rank_list):
            continue
            
        # Get next program choice
        program = rank_list[next_choices[applicant]]
        next_choices[applicant] += 1
        
        # If program has space
        if len(matches[program]) < program_capacities[program]:
            matches[program].append(applicant)
        else:
            # Compare with current matches
            current_matches = matches[program]
            ranked_applicants = program_preferences[program]
            
            # Get rankings of current matches and new applicant
            rankings = {app: ranked_applicants.index(app) 
                       if app in ranked_applicants else float('inf') 
                       for app in current_matches + [applicant]}
            
            # Sort by ranking
            sorted_applicants = sorted(rankings.keys(), key=lambda x: rankings[x])
            
            # Take top applicants up to capacity
            capacity = program_capacities[program]
            accepted = sorted_applicants[:capacity]
            
            if applicant in accepted:
                # Find who to remove
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
    
    # Calculate unfilled positions
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