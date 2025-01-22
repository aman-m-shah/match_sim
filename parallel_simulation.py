import multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Callable
import numpy as np
import random
from functools import partial
import copy
from models import Applicant, Program, Specialty, ScoreWeights, MatchResult
from simulation import generate_other_applicants, generate_applicant_preferences
from matcher import calculate_applicant_score, generate_rank_order, run_matching_round

def run_single_simulation(
    our_applicant: Applicant,
    programs: Dict[str, Program],
    specialties: Dict[str, Specialty],
    weights: ScoreWeights,
    seed: int
) -> Tuple[str, float, int]:
    """Run a single simulation with the given seed."""
    # Set random seed for reproducibility
    np.random.seed(seed)
    random.seed(seed)
    
    # Create deep copies to avoid shared state issues
    local_programs = copy.deepcopy(programs)
    local_specialties = copy.deepcopy(specialties)
    
    # Generate other applicants
    other_applicants = generate_other_applicants(local_programs, local_specialties, weights)
    all_applicants = {'our_applicant': our_applicant, **other_applicants}
    
    # Generate preferences
    applicant_prefs = generate_applicant_preferences(
        all_applicants, local_programs, local_specialties, weights
    )
    
    # Generate program preferences
    program_prefs = {
        prog_name: generate_rank_order(
            all_applicants, program, local_specialties[program.specialty], weights
        )
        for prog_name, program in local_programs.items()
    }
    
    # Run the match
    result = run_matching_round(
        applicant_prefs,
        program_prefs,
        {p.name: p.spots_available for p in local_programs.values()},
        weights
    )
    
    # Get match result for our applicant
    matched_program = result.applicant_matches.get('our_applicant', 'No Match')
    
    if matched_program != 'No Match':
        program = local_programs[matched_program]
        specialty = local_specialties[program.specialty]
        score = calculate_applicant_score(our_applicant, program, specialty, weights)
        rank = our_applicant.rank_list.index(matched_program) + 1
    else:
        score = 0.0
        rank = -1
    
    return matched_program, score, rank

def parallel_simulate_match(
    our_applicant: Applicant,
    programs: Dict[str, Program],
    specialties: Dict[str, Specialty],
    weights: ScoreWeights,
    n_simulations: int = 1000,
    n_processes: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    """Run match simulations in parallel."""
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)  # Leave one CPU free
    
    # Create pool of workers
    pool = mp.Pool(processes=n_processes)
    
    # Prepare simulation function with fixed arguments
    sim_func = partial(
        run_single_simulation,
        our_applicant,
        programs,
        specialties,
        weights
    )
    
    # Generate seeds for reproducibility
    seeds = list(range(n_simulations))
    
    # Initialize counters
    match_counts = {prog: 0 for prog in programs}
    match_counts['No Match'] = 0
    
    detailed_stats = {
        prog: {
            'scores': [],
            'ranks': []
        } for prog in programs
    }
    
    # Run simulations in chunks for progress updates
    chunk_size = max(1, n_simulations // 100)  # Update progress roughly every 1%
    completed = 0
    
    try:
        # Process results in chunks
        for chunk_results in pool.imap_unordered(sim_func, seeds, chunksize=chunk_size):
            matched_program, score, rank = chunk_results
            
            # Update counts and stats
            match_counts[matched_program] += 1
            if matched_program != 'No Match':
                detailed_stats[matched_program]['scores'].append(score)
                detailed_stats[matched_program]['ranks'].append(rank)
            
            # Update progress
            completed += 1
            if progress_callback:
                progress_callback(completed, n_simulations)
    
    finally:
        pool.close()
        pool.join()
    
    # Calculate probabilities
    probabilities = {
        name: count/n_simulations 
        for name, count in match_counts.items()
    }
    
    # Calculate final statistics
    final_stats = {}
    for prog in detailed_stats:
        scores = detailed_stats[prog]['scores']
        ranks = detailed_stats[prog]['ranks']
        if scores:
            final_stats[prog] = {
                'avg_score': float(np.mean(scores)),
                'std_score': float(np.std(scores)) if len(scores) > 1 else 0.0,
                'avg_rank': float(np.mean(ranks)),
                'std_rank': float(np.std(ranks)) if len(ranks) > 1 else 0.0,
                'n_matches': len(scores),
                'match_rate': float(len(scores)) / n_simulations
            }
        else:
            final_stats[prog] = {
                'avg_score': 0.0,
                'std_score': 0.0,
                'avg_rank': 0.0,
                'std_rank': 0.0,
                'n_matches': 0,
                'match_rate': 0.0
            }
    
    return probabilities, final_stats