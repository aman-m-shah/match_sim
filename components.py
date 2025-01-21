import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from models import (
    Specialty, Program, Applicant, ScoreWeights, 
    ExamScores, MatchResult
)
from typing import List, Dict, Optional, Tuple

def score_weights_sidebar():
    """Create sidebar interface for adjusting score weights."""
    st.sidebar.header("Scoring Weights")
    
    weights = ScoreWeights()
    
    with st.sidebar.expander("Base Score Weights", expanded=False):
        # Core score components
        weights.step2_weight = st.slider(
            "STEP 2 CK Weight",
            0.0, 0.5, 0.25,
            help="Weight of STEP 2 CK score in base calculation"
        )
        weights.comlex2_weight = st.slider(
            "COMLEX Level 2 Weight",
            0.0, 0.5, 0.25,
            help="Weight of COMLEX Level 2 score in base calculation"
        )
        weights.interview_weight = st.slider(
            "Interview Weight",
            0.0, 0.5, 0.30,
            help="Weight of interview performance"
        )
        weights.letters_weight = st.slider(
            "Letters Weight",
            0.0, 0.5, 0.15,
            help="Weight of letters of recommendation"
        )
        weights.research_weight = st.slider(
            "Research Weight",
            0.0, 0.5, 0.15,
            help="Weight of research publications"
        )
        
        base_sum = (
            weights.step2_weight +
            weights.comlex2_weight +
            weights.interview_weight +
            weights.letters_weight +
            weights.research_weight
        )
        st.info(f"Base weights sum: {base_sum:.2f} (should be close to 1.0)")
    
    with st.sidebar.expander("Bonus Weights", expanded=False):
        # Bonus factors
        weights.research_alignment_bonus = st.slider(
            "Research Alignment Bonus",
            0.0, 0.3, 0.10,
            help="Additional bonus for research alignment"
        )
        weights.away_rotation_bonus = st.slider(
            "Away Rotation Bonus",
            0.0, 0.3, 0.15,
            help="Additional bonus for away rotation"
        )
        weights.alumni_bonus = st.slider(
            "Alumni Connection Bonus",
            0.0, 0.3, 0.05,
            help="Additional bonus for alumni connection"
        )
        weights.geographic_bonus = st.slider(
            "Geographic Preference Bonus",
            0.0, 0.3, 0.10,
            help="Additional bonus for geographic preference"
        )
    
    with st.sidebar.expander("Program Tier Factors", expanded=False):
        # Tier adjustment factors
        weights.tier_1_factor = st.slider(
            "Tier 1 Program Factor",
            0.8, 1.0, 1.0,
            help="Score multiplier for tier 1 programs"
        )
        weights.tier_2_factor = st.slider(
            "Tier 2 Program Factor",
            0.7, 0.95, 0.9,
            help="Score multiplier for tier 2 programs"
        )
        weights.tier_3_factor = st.slider(
            "Tier 3 Program Factor",
            0.6, 0.9, 0.8,
            help="Score multiplier for tier 3 programs"
        )
    
    with st.sidebar.expander("Random Factors", expanded=False):
        # Randomization factors
        weights.score_random_factor = st.slider(
            "Score Random Factor",
            0.0, 0.5, 0.20,
            help="Random variation in initial score calculation"
        )
        weights.rank_random_factor = st.slider(
            "Rank List Random Factor",
            0.0, 0.5, 0.15,
            help="Random variation in rank list generation"
        )
        weights.match_random_factor = st.slider(
            "Match Algorithm Random Factor",
            0.0, 0.5, 0.10,
            help="Random variation in match algorithm"
        )
    
    # Validate weights
    is_valid, message = weights.validate()
    if not is_valid:
        st.sidebar.error(message)
    else:
        st.sidebar.success("Weight configuration is valid")
    
    return weights
def specialty_management():
    """Handle specialty creation and display."""
    st.header("Specialty Details")
    with st.form("specialty_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            specialty_name = st.text_input("Specialty Name")
            total_applicants = st.number_input("Total Applicants", min_value=1, value=100)
            total_programs = st.number_input("Total Programs", min_value=1, value=20)
            
        with col2:
            competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5,
                                     help="How competitive is this specialty (0=least, 1=most)")
            research_emphasis = st.slider("Research Emphasis", 0.0, 1.0, 0.5,
                                       help="How important is research (0=least, 1=most)")
            avg_interviews = st.number_input("Avg Interviews/Applicant", min_value=1, value=12)
            
        with col3:
            avg_spots = st.number_input("Avg Spots/Program", min_value=1, value=5)
            step2_min = st.number_input("Minimum STEP 2 Score", min_value=0, value=220)
            comlex2_min = st.number_input("Minimum COMLEX 2 Score", min_value=0, value=450)
        
        if st.form_submit_button("Add Specialty"):
            if specialty_name:
                st.session_state.specialties[specialty_name] = Specialty(
                    name=specialty_name,
                    total_applicants=total_applicants,
                    total_programs=total_programs,
                    competitiveness_factor=competitiveness,
                    research_emphasis=research_emphasis,
                    avg_interviews_per_applicant=avg_interviews,
                    avg_spots_per_program=avg_spots,
                    step2_min=step2_min,
                    comlex2_min=comlex2_min
                )

def display_specialties():
    """Display current specialties table."""
    if st.session_state.specialties:
        st.subheader("Current Specialties")
        specialty_df = pd.DataFrame([
            {
                'Specialty': s.name,
                'Total Programs': s.total_programs,
                'Total Applicants': s.total_applicants,
                'Avg Spots': s.avg_spots_per_program,
                'Avg Interviews': s.avg_interviews_per_applicant,
                'Competitiveness': f"{s.competitiveness_factor:.2f}",
                'Research Emphasis': f"{s.research_emphasis:.2f}",
                'Min STEP 2': s.step2_min,
                'Min COMLEX 2': s.comlex2_min,
                'Interview Rate': f"{s.interview_rate:.1%}"
            }
            for s in st.session_state.specialties.values()
        ])
        st.dataframe(specialty_df)

def program_entry_form():
    """Create program entry form."""
    st.header("Program Details")
    with st.form("program_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input("Program Name")
            specialty = st.selectbox("Specialty", options=list(st.session_state.specialties.keys()))
            rank = st.number_input("Your Rank", min_value=1, max_value=20)
            spots = st.number_input("Available Spots", min_value=1, value=5)
            
        with col2:
            program_tier = st.selectbox("Program Tier", options=[1, 2, 3],
                                      help="1=Top tier, 3=Lower tier")
            interview_score = st.slider("Your Interview Performance", 0.0, 1.0, 0.5)
            letter_score = st.slider("Your Letter Strength", 0.0, 1.0, 0.5)
            publications = st.number_input("Your Publications", 0, 10)
            
        with col3:
            is_do_friendly = st.checkbox("DO Friendly Program")
            research_focus = st.checkbox("Research-Focused Program")
            away_rotation = st.checkbox("Completed Away Rotation")
            alumni_connection = st.checkbox("Alumni Connection")
            geographic_pref = st.checkbox("Geographic Preference")
        
        if st.form_submit_button("Add Program"):
            if name and specialty in st.session_state.specialties:
                program = Program(
                    name=name,
                    specialty=specialty,
                    rank=rank,
                    spots_available=spots,
                    program_tier=program_tier,
                    research_focus=research_focus,
                    interview_performance=interview_score,
                    letter_strength=letter_score,
                    research_publications=publications,
                    geographic_preference=geographic_pref,
                    away_rotation=away_rotation,
                    alumni_connection=alumni_connection,
                    is_do_friendly=is_do_friendly
                )
                return program
    return None

def display_program_list():
    """Display editable program list."""
    if st.session_state.programs:
        st.subheader("Your Rank List")
        programs_df = pd.DataFrame([
            {
                'Program': p.name,
                'Specialty': p.specialty,
                'Rank': p.rank,
                'Program Tier': p.program_tier,
                'Spots': p.spots_available,
                'Interview Score': f"{p.interview_performance:.2f}",
                'Letter Score': f"{p.letter_strength:.2f}",
                'Publications': p.research_publications,
                'DO Friendly': p.is_do_friendly,
                'Research Focus': p.research_focus,
                'Away Rotation': p.away_rotation,
                'Alumni': p.alumni_connection,
                'Geographic': p.geographic_preference
            }
            for p in st.session_state.programs.values()
        ]).sort_values('Rank')

        edited_df = st.data_editor(
            programs_df,
            column_config={
                "DO Friendly": st.column_config.CheckboxColumn(
                    "DO Friendly",
                    default=False,
                ),
                "Research Focus": st.column_config.CheckboxColumn(
                    "Research Focus",
                    default=False,
                ),
                "Away Rotation": st.column_config.CheckboxColumn(
                    "Away Rotation",
                    default=False,
                ),
                "Alumni": st.column_config.CheckboxColumn(
                    "Alumni",
                    default=False,
                ),
                "Geographic": st.column_config.CheckboxColumn(
                    "Geographic",
                    default=False,
                ),
                "Specialty": st.column_config.SelectboxColumn(
                    "Specialty",
                    options=list(st.session_state.specialties.keys()),
                ),
                "Program Tier": st.column_config.SelectboxColumn(
                    "Program Tier",
                    options=[1, 2, 3],
                ),
            },
            disabled=["Program"],
            hide_index=True,
        )
        return edited_df
    return None

def display_match_results(probabilities: Dict[str, float], detailed_stats: Dict[str, Dict]):
    """Display match simulation results."""
    # Program bar chart
    st.subheader("Match Probabilities by Program")
    prob_df = pd.DataFrame([
        {
            'Program': prog,
            'Probability': prob,
            'Average Score': detailed_stats[prog].get('avg_score', 0) if prog != 'No Match' else 0,
            'Average Rank': detailed_stats[prog].get('avg_rank', 0) if prog != 'No Match' else 0
        }
        for prog, prob in probabilities.items()
    ]).sort_values('Probability', ascending=False)

    fig = go.Figure(data=[
        go.Bar(
            x=prob_df['Program'],
            y=prob_df['Probability'],
            text=prob_df['Probability'].apply(lambda x: f'{x:.1%}'),
            textposition='outside',
            hovertemplate="<b>%{x}</b><br>" +
                         "Probability: %{text}<br>" +
                         "Avg Score: %{customdata[0]:.2f}<br>" +
                         "Avg Rank: %{customdata[1]:.1f}<br>" +
                         "<extra></extra>",
            customdata=prob_df[['Average Score', 'Average Rank']].values
        )
    ])
    
    fig.update_layout(
        title="Match Probability by Program",
        xaxis_title="Program",
        yaxis_title="Probability",
        yaxis_tickformat='.1%'
    )
    st.plotly_chart(fig)

    # Summary statistics
    st.subheader("Summary Statistics")
    total_match_prob = 1 - probabilities.get('No Match', 0)
    stats = {
        'Overall Match Probability': f"{total_match_prob:.1%}",
        'Most Likely Program': prob_df.loc[prob_df['Program'] != 'No Match', 'Program'].iloc[0],
        'Average Score': f"{prob_df['Average Score'].mean():.2f}",
        'Total Programs': len(st.session_state.programs),
        'Number of Specialties': len(set(p.specialty for p in st.session_state.programs.values()))
    }
    st.table(pd.DataFrame([stats]).T.rename(columns={0: 'Value'}))