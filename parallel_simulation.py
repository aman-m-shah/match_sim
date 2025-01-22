import multiprocessing as mp
from typing import Dict, List, Tuple, Optional
import numpy as np
import random
from functools import partial
import copy
import signal
from contextlib import contextmanager
from models import Applicant, Program, Specialty, MatchResult
from simulation import generate_other_applicants, generate_applicant_preferences
from matcher import generate_rank_order, run_matching_round, calculate_applicant_score

class TimeoutException(Exception): 
    pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Simulation timed out")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def run_single_simulation(seed: int, our_applicant: Applicant, programs: Dict[str, Program], 
                         specialties: Dict[str, Specialty]) -> Tuple[str, float, int]:
    try:
        with time_limit(30):
            np.random.seed(seed)
            random.seed(seed)
            
            local_programs = copy.deepcopy(programs)
            local_specialties = copy.deepcopy(specialties)
            
            other_applicants = generate_other_applicants(local_programs, local_specialties)
            all_applicants = {'our_applicant': our_applicant, **other_applicants}
            
            applicant_prefs = generate_applicant_preferences(
                all_applicants, local_programs, local_specialties
            )
            
            program_prefs = {
                prog_name: generate_rank_order(
                    all_applicants, program, local_specialties[program.specialty]
                )
                for prog_name, program in local_programs.items()
            }
            
            result = run_matching_round(
                applicant_prefs,
                program_prefs,
                {p.name: p.spots_available for p in local_programs.values()}
            )
            
            matched_program = result.applicant_matches.get('our_applicant', 'No Match')
            
            if matched_program != 'No Match':
                program = local_programs[matched_program]
                specialty = local_specialties[program.specialty]
                score = calculate_applicant_score(our_applicant, program, specialty)
                rank = our_applicant.rank_list.index(matched_program) + 1
                return matched_program, score, rank
            
            return 'No Match', 0.0, -1
            
    except Exception as e:
        print(f"Simulation failed with error: {str(e)}")
        return 'No Match', 0.0, -1

def parallel_simulate_match(
    our_applicant: Applicant,
    programs: Dict[str, Program],
    specialties: Dict[str, Specialty],
    n_simulations: int = 1000,
    n_processes: Optional[int] = None,
    progress_callback = None
) -> Tuple[Dict[str, float], Dict[str, Dict]]:
    
    if n_processes is None:
        n_processes = max(1, mp.cpu_count() - 1)
    
    pool = None
    match_counts = {prog: 0 for prog in programs}
    match_counts['No Match'] = 0
    detailed_stats = {prog: {'scores': [], 'ranks': []} for prog in programs}
    completed = 0
    
    try:
        pool = mp.Pool(processes=n_processes)
        sim_func = partial(run_single_simulation, our_applicant=our_applicant, 
                         programs=programs, specialties=specialties)
        
        chunk_size = max(1, n_simulations // 100)
        for result in pool.imap_unordered(sim_func, range(n_simulations), chunksize=chunk_size):
            matched_program, score, rank = result
            
            match_counts[matched_program] += 1
            if matched_program != 'No Match':
                detailed_stats[matched_program]['scores'].append(score)
                detailed_stats[matched_program]['ranks'].append(rank)
            
            completed += 1
            if progress_callback:
                progress_callback(completed, n_simulations)
    
    except Exception as e:
        print(f"Parallel simulation failed: {str(e)}")
        raise e
    
    finally:
        if pool:
            pool.close()
            pool.terminate()
            pool.join()
    
    probabilities = {name: count/n_simulations for name, count in match_counts.items()}
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
                'n_matches': len(scores)
            }
        else:
            final_stats[prog] = {
                'avg_score': 0.0, 'std_score': 0.0,
                'avg_rank': 0.0, 'std_rank': 0.0,
                'n_matches': 0
            }
    
    return probabilities, final_stats