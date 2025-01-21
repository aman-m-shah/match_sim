import streamlit as st
import numpy as np
from models import ScoreWeights, ExamScores, Applicant
from simulation import simulate_match
from components import (
    score_weights_sidebar,
    specialty_management,
    display_specialties,
    program_entry_form,
    display_program_list,
    display_match_results
)

def initialize_session_state():
    """Initialize session state variables."""
    if 'specialties' not in st.session_state:
        st.session_state.specialties = {}
    if 'programs' not in st.session_state:
        st.session_state.programs = {}
    if 'score_weights' not in st.session_state:
        st.session_state.score_weights = ScoreWeights()
    if 'applicant_type' not in st.session_state:
        st.session_state.applicant_type = 'MD'
    if 'exam_scores' not in st.session_state:
        st.session_state.exam_scores = ExamScores()

def applicant_setup():
    """Setup applicant details."""
    st.sidebar.header("Applicant Details")
    
    # Applicant type selection
    applicant_type = st.sidebar.radio(
        "Applicant Type",
        options=['MD', 'DO'],
        help="Select your degree type"
    )
    st.session_state.applicant_type = applicant_type
    
    # Exam score inputs
    st.sidebar.subheader("Exam Scores")
    if applicant_type == 'MD':
        step2_score = st.sidebar.number_input(
            "STEP 2 CK Score",
            min_value=0,
            max_value=300,
            value=240,
            help="Enter your STEP 2 CK score"
        )
        comlex2_score = None
    else:  # DO student
        col1, col2 = st.sidebar.columns(2)
        with col1:
            step2_score = st.number_input(
                "STEP 2 CK Score (optional)",
                min_value=0,
                max_value=300,
                value=0,
                help="Enter your STEP 2 CK score if taken"
            )
        with col2:
            comlex2_score = st.number_input(
                "COMLEX Level 2 Score",
                min_value=0,
                max_value=800,
                value=500,
                help="Enter your COMLEX Level 2 score"
            )
    
    # Update exam scores in session state
    st.session_state.exam_scores = ExamScores(
        step2_ck=step2_score if step2_score > 0 else None,
        comlex2=comlex2_score if comlex2_score and comlex2_score > 0 else None
    )
    
    # Additional applicant details
    st.sidebar.subheader("Additional Details")
    st.session_state.research_pubs = st.sidebar.number_input(
        "Research Publications",
        min_value=0,
        max_value=20,
        value=0,
        help="Number of research publications"
    )
    
    st.session_state.research_focus = st.sidebar.checkbox(
        "Research Focus",
        help="Check if you have significant research experience"
    )

def create_applicant(rank_list):
    """Create applicant object from session state and inputs."""
    return Applicant(
        name='our_applicant',
        rank_list=rank_list,
        interview_score=0.75,  # Could make configurable
        exam_scores=st.session_state.exam_scores,
        letter_scores=0.75,  # Could make configurable
        publications=st.session_state.research_pubs,
        is_do=(st.session_state.applicant_type == 'DO'),
        research_focus=st.session_state.research_focus,
        away_rotation=True,  # Could make configurable
        alumni_connection=False  # Could make configurable
    )

def main():
    st.title("Residency Match Probability Calculator")
    
    # Initialize session state
    initialize_session_state()
    
    # Sidebar setup
    applicant_setup()
    
    # Settings sidebar
    with st.sidebar:
        n_simulations = st.number_input(
            "Number of Simulations",
            min_value=100,
            max_value=10000,
            value=1000,
            step=100,
            help="More simulations give more accurate results but take longer to run"
        )
        
        # Show national averages
        with st.expander("National Averages"):
            averages = ExamScores.get_national_averages()
            st.write("STEP 2 CK:")
            st.write(f"Mean: {averages['step2_ck']['mean']}")
            st.write(f"Std Dev: {averages['step2_ck']['std']}")
            st.write("COMLEX Level 2:")
            st.write(f"Mean: {averages['comlex2']['mean']}")
            st.write(f"Std Dev: {averages['comlex2']['std']}")
    
    # Main content area
    tabs = st.tabs(["Setup Programs", "Run Simulation", "Analysis"])
    
    with tabs[0]:
        # Specialty Management
        specialty_management()
        display_specialties()
        
        # Program Management
        new_program = program_entry_form()
        if new_program:
            st.session_state.programs[new_program.name] = new_program
        
        display_program_list()
        
        # Clear All button
        if st.button("Clear All"):
            st.session_state.programs = {}
            st.session_state.specialties = {}
            st.experimental_rerun()
    
    with tabs[1]:
        if not st.session_state.programs:
            st.warning("Please add programs in the Setup tab first.")
            return
        
        if st.button("Run Match Simulation"):
            with st.spinner("Running simulations..."):
                # Validate exam scores
                if st.session_state.applicant_type == 'DO' and not st.session_state.exam_scores.comlex2:
                    st.error("COMLEX Level 2 score is required for DO applicants.")
                    return
                if not st.session_state.exam_scores.step2_ck and not st.session_state.exam_scores.comlex2:
                    st.error("At least one exam score is required.")
                    return
                
                # Create rank list from programs
                rank_list = [p.name for p in sorted(
                    st.session_state.programs.values(),
                    key=lambda x: x.rank
                )]
                
                # Create applicant
                our_applicant = create_applicant(rank_list)
                
                # Create progress bar
                progress_text = "Running simulations..."
                progress_bar = st.progress(0, text=progress_text)
                status_container = st.empty()
                
                def update_progress(current, total):
                    progress = float(current) / float(total)
                    progress_bar.progress(
                        progress,
                        text=f"{progress_text} ({current}/{total})"
                    )
                    status_container.text(f"Completed {current} of {total} simulations")
                
                try:
                    # Run simulation
                    probabilities, detailed_stats = simulate_match(
                        our_applicant,
                        st.session_state.programs,
                        st.session_state.specialties,
                        n_simulations,
                        progress_callback=update_progress
                    )
                    
                    # Clear progress indicators
                    progress_bar.empty()
                    status_container.empty()
                    
                    # Display results
                    display_match_results(probabilities, detailed_stats)
                    
                except Exception as e:
                    progress_bar.empty()
                    status_container.empty()
                    st.error(f"An error occurred: {str(e)}")
                    raise e
    
    with tabs[2]:  # Changed from tabs[3] to tabs[2]
        # Analysis tab
        st.subheader("Analysis")
        if not st.session_state.programs:
            st.warning("Please add programs and run simulation first.")
            return
            
        # Add analysis features here
        st.write("Analysis features coming soon!")
        st.write("- Specialty-specific insights")
        st.write("- Program comparison tools")
        st.write("- Strategy recommendations")

if __name__ == "__main__":
    main()