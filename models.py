from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class ScoreWeights:
    """Weights for different components in applicant scoring."""
    random_factor: float = 0.2
    interview: float = 0.25
    letters: float = 0.15
    program_fit: float = 0.2
    research: float = 0.15
    geographic: float = 0.1
    rotation: float = 0.1
    alumni: float = 0.05
    top_candidate: float = 0.15
    step2_weight: float = 0.3
    comlex2_weight: float = 0.3

@dataclass
class ExamScores:
    """Container for standardized exam scores."""
    step2_ck: Optional[int] = None
    comlex2: Optional[int] = None

    @staticmethod
    def get_national_averages() -> Dict[str, Dict[str, float]]:
        """Return national average scores and standard deviations."""
        return {
            'step2_ck': {
                'mean': 245,
                'std': 15
            },
            'comlex2': {
                'mean': 500,
                'std': 85
            }
        }
    
    def normalize_scores(self) -> Dict[str, float]:
        """Normalize scores to 0-1 scale based on national averages."""
        averages = self.get_national_averages()
        normalized = {}
        
        if self.step2_ck is not None:
            z_score = (self.step2_ck - averages['step2_ck']['mean']) / averages['step2_ck']['std']
            normalized['step2_ck'] = min(1.0, max(0.0, 0.5 + (z_score * 0.15)))
        
        if self.comlex2 is not None:
            z_score = (self.comlex2 - averages['comlex2']['mean']) / averages['comlex2']['std']
            normalized['comlex2'] = min(1.0, max(0.0, 0.5 + (z_score * 0.15)))
        
        return normalized

@dataclass
class Specialty:
    """Represents a medical specialty."""
    name: str
    total_applicants: int
    total_programs: int
    competitiveness_factor: float  # Scale 0-1, higher is more competitive
    research_emphasis: float  # Scale 0-1, higher means more research-focused
    avg_interviews_per_applicant: int  # Average number of interviews given
    avg_spots_per_program: int  # Average number of spots per program in this specialty
    step2_min: Optional[int] = None  # Minimum Step 2 score typically accepted
    comlex2_min: Optional[int] = None  # Minimum COMLEX Level 2 score typically accepted
    
    @property
    def interview_rate(self) -> float:
        """Calculate interview rate based on applicants and programs."""
        return min(1.0, self.avg_interviews_per_applicant / self.total_programs)

@dataclass
class Program:
    """Represents a residency program."""
    name: str
    specialty: str
    rank: int  # Applicant's ranking of this program
    spots_available: int  # Number of positions available
    program_tier: int  # 1-3, where 1 is top tier
    research_focus: bool  # Whether program emphasizes research
    step2_min: Optional[int] = None  # Program-specific Step 2 minimum
    comlex2_min: Optional[int] = None  # Program-specific COMLEX minimum
    interview_performance: float = 0.5  # Applicant's interview performance (0-1)
    letter_strength: float = 0.5  # Strength of letters of recommendation (0-1)
    research_publications: int = 0  # Number of research publications
    geographic_preference: bool = False  # Whether applicant has geographic preference
    away_rotation: bool = False  # Whether applicant did away rotation
    alumni_connection: bool = False  # Whether applicant has alumni connection
    is_do_friendly: bool = True  # Whether program typically accepts DO students
    total_interviews: Optional[int] = None  # Total number of interviews conducted
    
    def __post_init__(self):
        if self.total_interviews is None:
            # Programs typically interview 10-12 candidates per spot
            self.total_interviews = self.spots_available * 10
        
        # Ensure minimum scores are set
        if not self.step2_min and not self.comlex2_min:
            self.step2_min = 220  # Default minimum Step 2 score
            self.comlex2_min = 450  # Default minimum COMLEX score

@dataclass
class Applicant:
    """Represents a residency applicant."""
    name: str
    rank_list: List[str]  # Programs where interviews completed, in rank order
    interview_score: float  # Overall interview performance (0-1)
    exam_scores: ExamScores  # STEP and COMLEX scores
    letter_scores: float  # Strength of letters (0-1)
    publications: int  # Number of publications
    is_do: bool  # Whether applicant is a DO student
    research_focus: bool = False  # Whether applicant emphasizes research
    away_rotation: bool = False  # Whether applicant did away rotation
    alumni_connection: bool = False  # Whether applicant has alumni connection

@dataclass
class MatchResult:
    """Contains the results of a match simulation."""
    applicant_matches: Dict[str, str]  # Applicant -> Program matches
    program_matches: Dict[str, List[str]]  # Program -> List[Applicant] matches
    unmatched_applicants: List[str]  # List of unmatched applicants
    unfilled_positions: Dict[str, int]  # Program -> number of unfilled spots
    
    @property
    def match_rate(self) -> float:
        """Calculate the overall match rate."""
        total_applicants = len(self.applicant_matches) + len(self.unmatched_applicants)
        return len(self.applicant_matches) / total_applicants if total_applicants > 0 else 0.0
    
    @property
    def fill_rate(self) -> float:
        """Calculate the overall program fill rate."""
        total_positions = sum(len(matches) for matches in self.program_matches.values())
        total_spots = total_positions + sum(self.unfilled_positions.values())
        return total_positions / total_spots if total_spots > 0 else 0.0