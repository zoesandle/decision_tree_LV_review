import streamlit as st
from typing import Dict, List
from PIL import Image
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from io import BytesIO

# ============================================================================
# TREE VISUALIZATION FUNCTION
# ============================================================================

NODE_MAPPING = {
    # Start
    'start': 'q_goal',
    
    # Construct branch
    'construct_focus': 'q_theory',
    'construct_theory_yes': 'q_theory',
    'construct_theory_no': 'q_allvar',
    'efa_continuous': 'efa',
    'efa_categorical': 'efa',
    'cfa_result': 'cfa',
    
    # People branch
    'people_focus': 'q_type',
    'lpa_path': 'q_long_cont',
    'lpa_result': 'lpa',
    'lca_path': 'q_long_cat',
    'lca_result': 'lca',
    
    # Relationship branch
    'relationship_focus': 'q_hyp',
    'sem_foundation': 'q_hyp',
    'sem_simple': 'q_effect',
    'sem_mediation': 'q_effect',
    'sem_moderation': 'q_effect',
    'sem_complex': 'q_effect',
    'sem_experimental': 'sem2',
    'sem_result': 'sem2',
    'network_analysis': 'network',
    
    # Time/Longitudinal branch
    'time_focus': 'q_tp',
    'latent_transition_result': 'lta',
    'growth_simple': 'q_groups',
    'growth_complex': 'q_groups',
    'lgcm_result': 'lcgm',
    'lgcm_mediation': 'q_groups',
    'lgcm_mediation_result': 'lcgm',
    'lgcm_predictors': 'q_groups',
    'lcga_result': 'lcga',
    'gmm_result': 'gmm',
    'growth_path2':'group_vars',
    'growth_path': 'q_groups'
}


def generate_tree_image(highlight_nodes=None):
    """
    Generate tree visualization
    
    Parameters:
    highlight_nodes: list of node keys to highlight in user's path
    """
    
    # ── colour palette ──────────────────────────────────────────────────────────
    C_DECISION = "#667eea"   # purple-100
    C_RESULT   = "#4caf50"   # green
    C_HIGHLIGHT = "#D9A300"  # gold for highlighted path
    C_EDGE_D   = "#534AB7"   # purple-600
    C_EDGE_R   = "#0F6E56"   # teal-600
    C_EDGE_A   = "#854F0B"   # amber-600
    C_TEXT     = "#ffffff"   # white
    C_ARROW    = "#E7DDDD"   # gray

    # ── node definitions  ────────────────────────────────────────────────────────
    nodes = {
        "q_goal":     ("What is your\nprimary goal?",        5.5, 13.5, 'd'),
        "q_theory":   ("Do you have a\ntheory about\nstructure?",          1.5, 12.9, 'd'),
        "q_allvar":   ("Do you want to\nmodel all variance?",  0.6, 11, 'd'),
        "pca":        ("PCA",                               0.1,  9.5, 'r'),
        "efa":        ("EFA / Bayesian EFA /\nIBP",          1.2,  8.6, 'r'),
        "q_sources":  ("Multiple data\nsources?",           2.8, 10.5, 'd'),
        "mtmm":       ("MTMM",                              2.2,  9.5, 'r'),
        "q_goal2":    ("What is your\ntheory about?",        3.2,  8.8, 'd'),
        "cfa":        ("CFA",                               1.5,  7.6, 'r'),
        "irt":        ("IRT",                               3.4,  7.6, 'r'),
        "sem1":       ("SEM",                               5.5,  7.6, 'r'),
        "q_type":     ("Continuous or\ncategorical variables?",   5.0, 12.5, 'd'),
        "q_long_cat": ("Longitudinal\ndata?",             4.3, 11.5, 'd'),
        "lca":        ("LCA",                               4.2,  10.5, 'r'),
        "lta":        ("LTA",                               5.9,  9.6, 'r'),
        "q_long_cont": ("Longitudinal\ndata?",               6.0, 11.6, 'd'),
        "lpa":        ("LPA",                               6.0,  10.7, 'r'),
        "q_hyp":      ("Exploratory or\nconfirm theory?",           10.1, 13.0, 'd'),
        "network":    ("Network\nAnalysis",                  10.9, 11.3, 'r'),
        "q_effect":   ("Linear or\nnon-linear\nrelationship?",      10.0, 10.7, 'd'),
        "sem2":       ("SEM",                               9.5,  9.6, 'r'),
        "comp":       ("Computational\nModeling",            11,  9.0, 'r'),
        "q_tp":       ("How many time\npoints?",            8, 12, 'd'),
        "q_groups":   ("Are you assuming\ngroups?",         8.1, 10.7, 'd'),
        "group_vars": ("Do groups have\ndifferent variances?",   8.8,  9.0, 'd'),
        "lcga":      ("Latent Class\nGrowth Analysis",       7.8,  8.0, 'r'),
        "lcgm":      ("Latent Growth\nCurve Modeling",       7.4,  8.8, 'r'),
        "gmm":       ("Growth\nMixture Modeling",            9.7,  7.9, 'r'),
    }

    # ── edges  (from, to, label) ─────────────────────────────────────────────────
    edges = [
        ("q_goal",     "q_theory",  "understand\nconstructs"),
        ("q_goal",     "q_type",    "understand\npeople"),
        ("q_goal",     "q_hyp",     "test\nrelationships"),
        ("q_goal",     "q_tp",      "change\nover time"),
        ("q_theory",   "q_sources", "yes"),
        ("q_theory",   "q_allvar",  "no theory"),
        ("q_sources",  "mtmm",      "yes"),
        ("q_sources",  "q_goal2",   "no"),
        ("q_goal2",    "cfa",       "construct structure"),
        ("q_goal2",    "irt",       "item influence"),
        ("q_goal2",    "sem1",      "relationships\nbetween constructs"),
        ("q_goal2",    "lcgm",      "change\nover\ntime"),
        ("q_allvar",   "pca",       "all variance"),
        ("q_allvar",   "efa",       "separate true\nfrom error"),
        ("q_type",     "q_long_cont",  "continuous"),
        ("q_type",     "q_long_cat",   "categorical"),
        ("q_long_cat",   "lca",       "no"),
        ("q_long_cont",   "q_tp",     "yes"),
        ("q_long_cont",   "lpa",     "no"),
        ("q_long_cat", "lta",       "yes"),
        ("q_hyp",      "q_effect",  "hypothesis"),
        ("q_hyp",      "network",   "explore"),
        ("q_effect",   "sem2",      "direct/\nmediation"),
        ("q_effect",   "comp",      "experimental\n+ moderation"),
        ("q_tp",           "lta",         "2 time\npoints"),
        ("q_tp",           "q_groups",    "3+ time\npoints"),
        ("q_groups",       "group_vars",  "yes"),
        ("group_vars",     "lcga",        "no"),
        ("group_vars",     "gmm",         "yes"),
        ("q_groups",       "lcgm",        "no groups"),
    ]

    # ── draw ─────────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 12))
    ax.set_xlim(-0.5, 11.7)  
    ax.set_ylim(7.0, 14)   
    ax.axis('off')
    fig.patch.set_facecolor('#0e1117')  # light blue background

    BOX_W, BOX_H = 1.1, 0.4

    def kind_color(k, node_key=None):
        if highlight_nodes and node_key in highlight_nodes:
            return C_HIGHLIGHT
        return {'d': C_DECISION, 'r': C_RESULT}[k]

    def kind_edge(k, node_key=None):
        if highlight_nodes and node_key in highlight_nodes:
            return "#FF8C00"  # orange for highlighted
        return {'d': C_EDGE_D, 'r': C_EDGE_R}[k]

    # draw nodes
    pos = {}
    for key, (label, x, y, kind) in nodes.items():
        pos[key] = (x, y)
        fc = kind_color(kind, key)
        ec = kind_edge(kind, key)
        linewidth = 2.5 if (highlight_nodes and key in highlight_nodes) else 1.2
        box = FancyBboxPatch(
            (x - BOX_W/2, y - BOX_H/2), BOX_W, BOX_H,
            boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor=ec, linewidth=linewidth, zorder=3
        )
        ax.add_patch(box)
        ax.text(x, y, label, ha='center', va='center',
                fontsize=12, color=C_TEXT, fontweight='600' if (highlight_nodes and key in highlight_nodes) else '500',
                zorder=4, multialignment='center')

    # draw edges
    for (src, dst, lbl) in edges:
        x1, y1 = pos[src]
        x2, y2 = pos[dst]

        dx, dy = x2 - x1, y2 - y1
        length = max(np.sqrt(dx**2 + dy**2), 1e-6)
        nx, ny = dx/length, dy/length

        ox = nx * BOX_W/2 * 1.05
        oy = ny * BOX_H/2 * 1.6

        sx, sy = x1 + ox, y1 + oy
        ex, ey = x2 - ox, y2 - oy

        # Highlight edges if both nodes are in path
        edge_color = C_ARROW
        edge_width = 0.9
        if highlight_nodes and src in highlight_nodes and dst in highlight_nodes:
            edge_color = "#FF8C00"
            edge_width = 2.5

        ax.annotate(
            "", xy=(ex, ey), xytext=(sx, sy),
            arrowprops=dict(
                arrowstyle="-|>",
                color=edge_color,
                lw=edge_width,
                mutation_scale=10,
                connectionstyle="arc3,rad=0.0"
            ),
            zorder=2
        )

        if lbl:
            mx, my = (sx + ex)/2, (sy + ey)/2
            perp_x = -ny * 0.18
            perp_y =  nx * 0.18
            ax.text(mx + perp_x, my + perp_y, lbl,
                    ha='center', va='center',
                    fontsize=12, color='#ffffff',
                    zorder=5, multialignment='center',
                    bbox=dict(facecolor='#0e1117', edgecolor='none',
                              alpha=0.75, pad=1.0))

    # legend
    legend_handles = [
        mpatches.Patch(facecolor=C_DECISION, edgecolor=C_EDGE_D, label='Decision node'),
        mpatches.Patch(facecolor=C_RESULT,   edgecolor=C_EDGE_R, label='Model result'),
    ]
    if highlight_nodes:
        legend_handles.append(
            mpatches.Patch(facecolor=C_HIGHLIGHT, edgecolor="#FF8C00", label='Your path')
        )
    ax.legend(handles=legend_handles, loc='lower left',
              fontsize=14, framealpha=0.9, edgecolor='#D3D1C7')

    ax.set_title("Latent Variable Model Decision Tree",
                 fontsize=20, fontweight='500', color=C_TEXT, pad=10)

    plt.tight_layout()
    
    # Return as bytes instead of saving
    img_buffer = BytesIO()
    plt.savefig(img_buffer, dpi=500, bbox_inches='tight', facecolor='#0e1117', format='png')
    img_buffer.seek(0)
    plt.close(fig)
    return img_buffer

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Latent Variable Model Selector",
    page_icon="🌳",
    layout="wide"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
    }
    .header h1 {
        margin: 0;
        font-size: 2em;
    }
    .breadcrumb {
        display: flex;
        gap: 10px;
        margin-bottom: 20px;
        flex-wrap: wrap;
    }
    .breadcrumb-item {
        background-color: #bcc2dc;
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 0.9em;
        border: 1px solid #667eea;
    }
    .breadcrumb-item.active {
        background-color: #667eea;
        color: white;
        border: 1px solid #667eea;
    }
    .result-box {
        background-color: #e8f5e9;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #4caf50;
        margin: 10px 0;
    }
    .result-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #2e7d32;
        margin-bottom: 10px;
    }
    .caution {
        background-color: #ff6262;
        padding: 6px;
        border-radius: 1px;
        border-left: 4px solid #d80000;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# DECISION TREE DATA
# ============================================================================
decision_tree = {
    "start": {
        "question": "What is your primary research goal?",
        "options": {
            "Understand constructs or reduce complexity": "construct_focus_path_theory",
            "Understand people": "people_focus",
            "Test relationships between variables": "relationship_focus",
            "Understand change over time": "time_focus"
        }
    },

    "construct_focus_path_theory": {
    "question": "Do you already have a theory about the structure of your construct?",
    "options": {
            "Yes, I have a theoretical model": "construct_focus_path_sources",
            "No, I want to explore the data": "construct_theory_no"
        }
    },

    "construct_focus_path_sources": {
        "question": "Do you have data from different sources?",
        "options": {
            "Yes, I have multiple types of data (e.g. self-report, behavioral, physiological)": "construct_focus_path_multi",
            "No, I just have one type of data (e.g. questionnaire responses)": "factors_or_components"
        }
    },

    "factors_or_components": {
        "question": "Do you want to model latent constructs (factors) or just reduce dimensionality (components)?",
        "options": {
            "Model latent constructs (separate error variance)": "construct_theory_yes",
            "Reduce dimensionality (capture all variance)": "pca_result"
        }
    },


    "construct_focus_path_multi": {
        "question": "Do want to model shared variance between ≥2 constructs and ≥ data sources?",
        "options": {
            "No, I want to test relationships but build my latent variables from multiple data sources": "sem_foundation",
            "Yes, I want to model overlap between constructs and data sources": "mtmm_result"
        }
    },

    "mtmm_result": {
        "result": "Multitrait-Multimethod (MTMM) Models",
        "recommendation": "Use MTMM models to model shared variance between constructs and data sources",
        "when_use": [
            "✓ You have multiple constructs measured with multiple methods",
            "✓ You want to separate trait variance from method variance",
            "✓ You want to test convergent and discriminant validity of your measures",
            "✓ Sample size 300+ depending on model complexity"
        ],
        "further_reading": ["Campbell, D. T., & Fiske, D. W. (1959). Convergent and discriminant validation by the multitrait-multimethod matrix. Psychological bulletin, 56(2), 81.",
                            "MTMM models can be done in Mplus and R (lavaan package).",
                            "✓ We have a study using MTMM modeling in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."],
    },


    "construct_theory_yes": {
        "question": "What do you want to do with your construct?",
        "options": {
            "Validate/test if my scale measures what I think": "cfa_result",
            "Investigate relationships between constructs": "relationship_focus",
            "Both validate AND test relationships": "sem_foundation",
            "Investigate developmental trajectories of my construct": "time_focus",
            "Investigate the influence of indivdual items on my construct": "irt_result"
        }
    },

    "irt_result": {
        "result": "Item Response Theory (IRT)",
        "recommendation": "Use IRT to model the relationship between latent traits and item responses, and to evaluate item properties",
        "when_use": [
            "✓ You want to validate a measurement scale",
            "✓ You want to understand how individual items relate to the underlying construct",
            "✓ You have categorical/ordinal item responses (e.g. Likert scales)",
        ],
        "further_reading": ["REFERENCE",
                            "IRT can be done in R (ltm package) and Mplus.",
                            "❌ We do not have a study using IRT in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."],
    },

    "construct_theory_no": {
        "question": "Do you want to model all variance or separate error variance from model structure?",
        "options": {
            "Model all variance": "pca_result",
            "Separate error variance ": "scale"
        }
    },

    "pca_result": {
        "result": "Principal Component Analysis (PCA)",
        "recommendation": "Use PCA to reduce dimensionality and identify components that explain variance",
        "when_use": [
            "✓ You want to reduce dimensionality",
            "✓ You have no theoretical model",
            "✓ You want to capture all variance (including error)",
            "✓ Sample size 100+"
        ],
        "further_reading": ["REFERENCE",
                            "You can do PCA in all available statistical software.",
                            "✓ We have studies using PCA in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find it."],
        "cautions": [
            "⚠ PCA is not a true latent variable model - it captures all variance, including error",
            "⚠ Components may not be interpretable as 'constructs'",
            "⚠ Consider EFA if you want to separate error variance"
        ]
    },

    "scale": {
        "question": "Are your variables continuous, ordinal, or categorical?",
        "options": {
            "Continuous (test scores, ratings)": "experimental_FA",
            "Ordinal (Likert scales, ordered categories)": "efa_ordinal",
            "Categorical/Binary (yes/no responses)": "efa_categorical"
        }
    },

    "experimental_FA": {
        "question": "Do you want a traditional approach or are you interested in trying newer methods?",
        "options": {
            "I want to use a traditional approach": "efa_continuous",
            "I want to quantify uncertainty in my estimates": "bayesian_efa",
            "I want to quantify uncertainty and circumvent the factor selection problem": "ibp_result"
        }
        },
    
    "bayesian_efa": {
        "result": "Bayesian Exploratory Factor Analysis",
        "recommendation": "Use Bayesian EFA to explore factor structure while quantifying uncertainty in estimates",
        "when_use": [
            "✓ You want to explore factor structure",
            "✓ You want credible intervals for factor loadings",
            "✓ You have continuous data (ratings, scores)",
            "✓ Sample size 100+"
        ],
        "further_reading": ["REFERENCE",
                            "Bayesian EFA can be done in R (blavaan package) and Mplus.",
                            "❌ We do not have any studies using Bayesian EFA in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."],
    },

    "ibp_result": {
        "result": "Indian Buffet Process (IBP)",
        "recommendation": "The IBP in a nonparametric bayesian method which characteizes structure in our data.",
        "when_use": [
            "✓ You want to explore factor structure without pre-specifying number of factors",
            "✓ You want to quantify uncertainty in factor loadings",
            "✓ You want to closely characterize individual differences in latent structure"
        ],
        "further_reading": ["Griffiths, T. L., & Ghahramani, Z. (2011). The Indian buffet process: An introduction and review. Journal of Machine Learning Research, 12(4).",
                            "IBP Factor Analysis is a newer method and may require custom implementation in R or Python.",
                            "✓ We have a study using the IBP in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."],
    },

    "efa_continuous": {
        "result": "Exploratory Factor Analysis (EFA)",
        "recommendation": "Use EFA to discover the underlying structure of your continuous variables",
        "when_use": [
            "✓ You have no theoretical structure",
            "✓ Variables are continuous (scales, ratings)",
            "✓ You want to explore what factors emerge",
            "✓ You may have >100 responses and multiple items"
        ],
        "next_step": "After EFA, consider CFA for validation in a new sample",
        "further reading": ["Fabrigar, L. R., & Wegener, D. T. (2012). Exploratory factor analysis. Oxford University Press.", #CHECK REFERENCE
                            "You can do EFA in all available statistical software.",
                            "✓ We have studies using EFA in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find it."]

    },

    "efa_ordinal": {
        "result": "Exploratory Factor Analysis (EFA) with Ordinal Data",
        "recommendation": "Use EFA with polychoric correlations to explore factor structure of ordinal variables",
        "when_use": [
            "✓ You have no theoretical structure",
            "✓ Variables are ordinal (Likert scales, ordered categories)",
            "✓ You want to explore what factors emerge",
            "✓ Sample size 200+ recommended"
        ],
        "next_step": "After EFA, consider CFA for validation in a new sample",
        "further_reading": ["Fabrigar, L. R., & Wegener, D. T. (2012). Exploratory factor analysis. Oxford University Press.",#ChecK REFERENCE
                            "You can do EFA with ordinal data in R (lavaan package) and Mplus.",
                            "✓ We have studies using EFA in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find it."]
    },

    "efa_categorical": {
        "result": "Exploratory Factor Analysis (EFA) with tetrachoric correlation",
        "recommendation": "Use EFA with probit or logit links to explore factor structure of categorical variables",
        "when_use": [
            "✓ Your variables are categorical (yes/no, true/false)",
            "✓ You want to explore latent structure",
            "✓ You have binary or ordinal responses",
            "✓ Sample size 300+"
        ],
        "caution": "⚠ EFA with categorical data is very rare. Consider doing a person centered approach instead."
    },

    "people_focus": {
        "question": "Are your variables continuous or categorical?",
        "options": {
            "Continuous (test scores, questionnaires, measurements)": "lpa_path",
            "Categorical (symptom presence/absence, binary responses)": "lca_path"
        }
    },

    "lpa_path": {
        "question": "Do you have multiple measurement time points (3+)?",
        "options": {
            "Yes, I have longitudinal data": "lcga_result",
            "No, just one time point": "lpa_result"
        }
    },

    "experimental_LPA": {
    "question": "Do you want a traditional approach or are you interested in trying newer methods?",
    "options": {
            "I want to use a traditional approach": "lpa_result",
            "I want to quantify uncertainty in my estimates": "bayesian_lpa",
            "I want to quantify uncertainty and circumvent the group selection problem": "dpm_lpa_result"
        }
        },


    "lpa_result": {
        "result": "Latent Profile Analysis (LPA)",
        "recommendation": "Use LPA to identify distinct profiles of people based on continuous measures",
        "when_use": [
            "✓ You want to identify subgroups/profiles",
            "✓ Variables are continuous (composites, scores)",
            "✓ You have cross-sectional data (one timepoint)",
            "✓ Sample size 500+ preferred",
            "✓ You care about how people DIFFER from each other"
        ],
        "further_reading": ["REFERENCE",
                            "You can do LPA in Mplus, R (tidyLPA package), and Latent GOLD.",
                            "✓ We have studies using LPA in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find them."
        ],

        "next_step": "Further research: Characterize profiles and test if they predict outcomes, for example with a follow-up SEM or regression model."
    },

    "bayesian_lpa": {
        "result": "Bayesian Latent Profile Analysis",
        "recommendation": "Use Bayesian LPA to identify profiles while quantifying uncertainty in class membership",
        "when_use": [
            "✓ You want to identify subgroups/profiles",
            "✓ You want posterior probabilities of class membership",
            "✓ You have continuous variables (composites, scores)",
            "✓ Sample size 500+ preferred"
        ],
        "further_reading": ["REFERENCE",
                            "Bayesian LPA can be done in R (blavaan package) and Mplus.",
                            "❌ We do not have any studies using Bayesian LPA in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."],
    },

    "dpm_lpa_result": {
        "result": "Dirichlet Process Mixture Model (DPM) for LPA",
        "recommendation": "Use DPM to identify latent profiles without pre-specifying number of classes",
        "when_use": [
            "✓ You want to identify subgroups/profiles",
            "✓ You want to avoid the class enumeration problem",
            "✓ You have continuous variables (composites, scores)",
            "✓ Sample size 500+ preferred"
        ],
        "further_reading": ["Paskewitz, S., Brazil, I. A., Yildirim, I., Ruiz, S., & Baskin-Sommers, A. (2024). Enhancing within-person estimation of neurocognition and the prediction of externalizing behaviors in adolescents. Computational Psychiatry, 8(1), 119.",
                            "DPM can be done in R (dpm package) and Mplus.",
                            "✓ We have a study using DPM in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."],
    },

    "lca_path": {
        "question": "Do you have multiple measurement time points (2+)?",
        "options": {
            "Yes, I have longitudinal data": "latent_transition_result",
            "No, just one time point": "lca_result"
        }
    },

    "lca_path2": {
        "question": "Do you want to identify how people transition between classes over time?",
        "options": {
            "Yes, I want to identify transitions": "latent_transition_result",
            "No, I want to measure how developmental trajectories differ between classes": "lcga_result",
            "No, I just want to identify classes at one time point": "lca_result"
        }
    },

    "latent_transition_result": {
        "result": "Latent Transition Analysis (LTA)",
        "recommendation": "Use LTA to identify latent classes and how individuals transition between them over time",
        "when_use": [
            "✓ You want to identify subgroups and their transitions",
            "✓ Variables are categorical (symptoms yes/no, diagnoses)",
            "✓ You have longitudinal data with at least 2 time points",
            "✓ Sample size 300+ minimum"
        ],
        "further_reading": ["REFERENCE",
                            "You can do LTA in Mplus and R (depmixS4 package).",
                            "✓ We have a study using LTA in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."],
    },

    "lca_result": {
        "result": "Latent Class Analysis (LCA)",
        "recommendation": "Use LCA to identify distinct classes from categorical indicators",
        "when_use": [
            "✓ You want to identify subgroups",
            "✓ Variables are categorical (symptoms yes/no, diagnoses)",
            "✓ You have cross-sectional data",
            "✓ Sample size 300+ minimum",
            "✓ You care about how people differ from each other"
        ],
        "further_reading": ["REFERENCE",
                            "You can do LCA in Mplus, R (poLCA package), and Latent GOLD.",
                            "✓ We have studies using LCA in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find them."
        ],
        "cautions": [
            "⚠ Be careful with class enumeration (3 vs 4 vs 5 classes?)",
        ],
        "next_step": "Further research: Characterize profiles and test if they predict outcomes, for example with a follow-up SEM or regression model."
    },

    "relationship_focus": {
        "question": "Do you have a hypothesis about the relationships between variables?",
        "question": "Do you want to test direct relationships, indirect/mediation effects, or both?",
        "details": "Direct = A→B | Indirect/mediation = A→B→C | Moderation = A→B (depends on C)",
        "options": {
            "I have a hypothesis about the relationships between variables": "sem_foundation",
            "I want to explore complex interrelationships without a specific model in mind": "network_analysis"
        }
    },

    "sem_foundation": {
        "question": "Do you want to test direct relationships, indirect/mediation effects, or both?",
        "details": "Direct = A→B | Indirect/mediation = A→B→C | Moderation = A→B (depends on C)",
        "options": {
            "Just direct relationships (A→B)": "sem_simple",
            "Indirect effects/mediation (A→B→C)": "sem_mediation",
            "Moderation/interaction (A→B depends on C)": "sem_moderation",
            "Complex combinations (multiple pathways)": "sem_complex"
        }
    },

    "sem_simple": {
        "question": "Do you have multiple measurement time points?",
        "options": {
            "Yes, 3+ time points": "lgcm_result",
            "No, just one time point": "sem_result"
        }
    },

    "sem_mediation": {
        "question": "Do you have multiple measurement time points?",
        "options": {
            "Yes, 3+ time points": "lgcm_mediation",
            "No, just one time point": "sem_result"
        }
    },

    "sem_moderation": {
        "question": "Do you have access to experimental data (random assignment)?",
        "options": {
            "Yes, experimental design": "computational_mod",
            "No, observational/correlational": "sem_result"
        }
    },

    "sem_complex": {
        "question": "Do you have multiple measurement time points?",
        "options": {
            "Yes, 3+ time points": "lgcm_result",
            "No, one time point": "sem_result"
        }
    },

    "computational_mod": {
        "result": "Computational Modeling",
        "recommendation": "Use computational modeling to test causal relationships with experimental design",
        "when_use": [
            "✓ You have experimental/random assignment",
            "✓ You can infer directionality from design",
            "✓ You want to test mechanistic models of behavior",
            "✓ Sample size depends on model complexity"
        ],
        "further_reading": [
            "REFERENCE",
            "Computational modeling can be done in with REFERENCE",
            "✓ We have a study using computational modeling in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."
        ]
    },

    "time_focus": {
        "question": "How many measurement time points do you have?",
        "options": {
            "3+ time points": "growth_path",
            "Only 2 time points": "latent_transition_result"
        }
    },

    "growth_path": {
        "question": "What kind of developmental patterns are you interested in?",
        "options": {
            "I want to identify how a construct changes over time": "lgcm_result",
            "I want to identify how people transition between classes over time": "latent_transition_result",
            "I want to identify subgroups with different developmental trajectories": "growth_path2"
        }
    },

    "growth_path2": {
        "question": "Do you believe your subgroups have different variances?",
        "options": {
            "No, I think they have similar variances": "lcga_result",
            "Yes, I think they have different variances": "gmm_result"
        }
    },

    "gmm_result": {
        "result": "Growth Mixture Modeling (GMM)",
        "recommendation": "Use GMM to identify subgroups with different developmental trajectories and variances",
        "when_use": [
            "✓ You have 3+ time points",
            "✓ You suspect distinct developmental pathways with different variances",
            "✓ Examples: 'starters' vs 'chronic' vs 'desisters'",
            "✓ Sample size 300-500+ (larger for more complex models)",
            "✓ You want to compare trajectory groups on outcomes"
        ],
        "further_reading": [
            "REFERENCE",
            "GMM can be done in Mplus, R (lcmm package), and Latent GOLD.",
            "❌ We do not have studies using GMM in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."
        ]
    },

    "lgcm_result": {
        "result": "Latent Growth Curve Modeling (LGCM)",
        "recommendation": "Use LGCM to model trajectories of change over time",
        "when_use": [
            "✓ You have 3+ time points",
            "✓ You want to model growth/change trajectories",
            "✓ You're interested in intercept (starting point) and slope (rate of change)",
            "✓ You want to predict trajectories from baseline variables",
            "✓ Sample size 100+",
            "⚠ Use a quadratic latent growth curve approach if your data show non-linear growth patterns"
        ],
        "further_reading": [
            "REFERENCE",
            "LGCM can be done in Mplus, R (lavaan package), and Latent GOLD.",
            "✓ We have studies using LGCM in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find them."
        ],
    },

    "lcga_result": {
        "result": "Latent Class Growth Analysis (LCGA) / Growth Mixture Modeling (GMM)",
        "recommendation": "Use LCGA/GMM to identify groups following different developmental trajectories",
        "when_use": [
            "✓ You have 3+ time points",
            "✓ You suspect distinct developmental pathways",
            "✓ Examples: 'starters' vs 'chronic' vs 'desisters'",
            "✓ Sample size 300-500+ (larger for more complex models)",
            "✓ You want to compare trajectory groups on outcomes"
        ],
        "further_reading": [
            "reference",
            "LCGA/GMM can be done in Mplus, R (lcmm package), and Latent GOLD.",
            "✓ We have studies using LCGA in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find them."
        ]
    },

    "lgcm_mediation": {
        "question": "Do you want to test mediation or just growth?",
        "options": {
            "Just model how change happens": "lgcm_result",
            "Test if variable A predicts growth through B": "lgcm_mediation_result",
            "Both growth and trajectory groups": "lcga_result"
        }
    },

    "lgcm_mediation_result": {
        "result": "Longitudinal Structural Equation Modeling / Parallel Process Latent Growth Curve Modeling",
        "recommendation": "Model growth in multiple variables and their relationships over time",
        "when_use": [
            "✓ You have 3+ time points",
            "✓ You want to model growth in multiple variables",
            "✓ Test if growth in one variable predicts growth in another",
            "✓ Example: Does EF growth predict aggression reduction?"
        ]
    },


    "cfa_result": {
        "result": "Confirmatory Factor Analysis (CFA)",
        "recommendation": "Use CFA to confirm a hypothesized factor structure of your construct.",
        "when_use": [
            "✓ You have a theoretical model of item structure",
            "✓ You want to validate a measurement scale",
            "✓ You have continuous data (scale responses, scores)",
            "✓ Sample size 200-300+ depending on complexity",
            "✓ You care about HOW variables relate (measurement quality)"
        ],
        "further_reading": ["Brown, T. A., & Moore, M. T. (2012). Confirmatory factor analysis. Handbook of structural equation modeling, 361(2012), 379.",
                            "You can do CFA in all available statistical software.",
                            "✓ We have a study using CFA in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find it."],
    },

    "sem_result": {
        "result": "Structural Equation Modeling (SEM)",
        "recommendation": "Use SEM to test relationships between latent constructs. SEMs can have many different designs depending on your theory, so make sure to specify your model correctly.",
        "when_use": [
            "✓ You want to test relationships between constructs",
            "✓ You want to test mediation (A→B→C) or moderation (A→B depends on C)",
            "✓ You have multiple indicators of each construct",
            "✓ Sample size 300-400+",
        ],
        "further_reading": [
            "Hair Jr, J. F., Hult, G. T. M., Ringle, C. M., Sarstedt, M., Danks, N. P., & Ray, S. (2021). An introduction to structural equation modeling. In Partial least squares structural equation modeling (PLS-SEM) using R: a workbook (pp. 1-29). Cham: Springer International Publishing.",
            "SEMs can be done using Mplus, R (lavaan package), and REFERENCE",
            "✓ We have studies using SEM in our review! Use our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/) to find them."
        ]
    },

    "network_analysis": {
        "result": "Network Analysis",
        "recommendation": "Use network analysis to model relationships between variables. This approach is mostly done using observed variables, but you can also test for network structure between different latent constructs or latent and observed variables.",
        "when_use": [
            "✓ You want to explore complex interrelationships",
            "✓ You have many variables",
            "✓ You want to identify central nodes in the network",
        ],
        "further_reading": [
            "REFERENCE",
            "Network analysis can be done in R (qgraph package) and Python (networkx library).",
            "✓ We have one study using network analysis in our [study explorer](https://studyexplorer-latentvariables-externalizing.streamlit.app/)."
        ]
    }
}

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if "current_node" not in st.session_state:
    st.session_state.current_node = "start"
    st.session_state.history = []
    st.session_state.show_description = False
    st.session_state.show_full_tree = False
    st.session_state.show_tree_with_path = False


# ============================================================================
# HEADER
# ============================================================================
st.markdown("""
    <div class="header">
        <h1>Model Selector for Latent Variable Models</h1>
        <p>Find the right latent variable model for your research question(s).</p>
        <p>Please let me know if you have any questions: zoe.sandle@donders.ru.nl</p>
    </div>
    """, unsafe_allow_html=True)

if st.button("Show full decision tree", key="show_tree_btn"):
    st.session_state.show_full_tree = True
    tree_image = generate_tree_image()
    st.image(tree_image, use_column_width=True, caption="Full Decision Tree for Selecting Latent Variable Models")
    if st.button("Close Tree", key="close_tree_path"):
        st.session_state.show_tree_with_path = False
        st.rerun()



# ============================================================================
# DISPLAY CURRENT NODE
# ============================================================================
current = decision_tree[st.session_state.current_node]

# Show description only on first page
if st.session_state.current_node == "start" and st.session_state.show_description:
    st.info(current.get("description", ""))
    st.write(current.get("details", ""))
    st.markdown("---")

# Show question or result
if "result" in current:
    # RESULT PAGE
    st.markdown(f"<div class='result-box'><div class='result-title'>✅ {current['result']}</div></div>", 
               unsafe_allow_html=True)
    
    st.markdown(f"**{current.get('recommendation', '')}**")
    
    if "when_use" in current:
        st.subheader("When to use this:")
        for item in current["when_use"]:
            st.write(item)

    if "further_reading" in current:
        st.subheader("Further reading:")
        for item in current["further_reading"]:
            st.write(f"- {item}")
    #turned off displaying cautions for now
    if "cautions" in current:
        st.subheader("Important cautions:")
        for caution in current["cautions"]:
            st.markdown(f"<div class='caution'>{caution}</div>", unsafe_allow_html=True)
    
    if "next_step" in current:
        st.info(f"**Next step:** {current['next_step']}")
    
    if "details" in current:
        st.write(current["details"])
    
    
    col1, col2, col3 = st.columns([1, 1, 1])
    # Back button
    with col1:
        if st.button("← Go Back", key="back_btn", use_container_width=True):
            if st.session_state.history:
                st.session_state.history.pop()
                if st.session_state.history:
                    prev_node = st.session_state.history[-1][0]
                else:
                    prev_node = "start"
                st.session_state.current_node = prev_node
                st.session_state.show_description = False
                st.rerun()

    
    with col2:
        if st.button("start over", key="restart_btn", use_container_width=True):
            st.session_state.current_node = "start"
            st.session_state.history = []
            st.session_state.show_description = True
            st.rerun()


# for now we are turning this off
#     
#    with col3:
 #       if st.button("Show decision tree with my path highlighted", key="show_path_btn", use_container_width=True):
  #          st.session_state.show_tree_with_path = True

   # if st.session_state.get("show_tree_with_path", False):
        # Map nodes from detailed tree to visualization nodes
    #    highlight_nodes = []
     #   for node, _ in st.session_state.history:
      #      viz_node = NODE_MAPPING.get(node, node)
       #     if viz_node not in highlight_nodes:
        #        highlight_nodes.append(viz_node)
        
        # Add current node
        #current_viz_node = NODE_MAPPING.get(st.session_state.current_node, st.session_state.current_node)
#        if current_viz_node not in highlight_nodes:
 #           highlight_nodes.append(current_viz_node)
        
        # Generate and display
  #      tree_image = generate_tree_image(highlight_nodes=highlight_nodes)
   #     st.image(tree_image, use_column_width=True, caption="Decision Tree with Your Path Highlighted")
        
    #    if st.button("Close Tree", key="close_tree_path"):
     #       st.session_state.show_tree_with_path = False
      #      st.rerun()

else:
    # QUESTION PAGE
    st.subheader(current.get("question", ""))
    
    if "details" in current:
        st.caption(current["details"])
    
    # Display options as buttons
    #st.markdown("**Choose an option:**")
    
    for idx, (option_text, next_node) in enumerate(current.get("options", {}).items()):
            if st.button(option_text, key=f"option_{idx}_{st.session_state.current_node}", use_container_width=True):
                # Update history
                st.session_state.history.append((st.session_state.current_node, option_text))
                # Move to next node
                st.session_state.current_node = next_node
                st.session_state.show_description = False
                st.rerun()
    
    # Back button for questions
    if st.session_state.history:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("← Go Back", key="back_question", use_container_width=True):
                if st.session_state.history:
                    st.session_state.history.pop()
                    if st.session_state.history:
                        prev_node = st.session_state.history[-1][0]
                    else:
                        prev_node = "start"
                    st.session_state.current_node = prev_node
                    st.session_state.show_description = False
                    st.rerun()
        with col2:
            if st.button("start over", key="restart_question", use_container_width=True):
                st.session_state.current_node = "start"
                st.session_state.history = []
                st.session_state.show_description = True
                st.rerun()

# ============================================================================
# NAVIGATION HISTORY (BREADCRUMB)
# ============================================================================
if st.session_state.history:
    st.markdown("<div class='breadcrumb'>", unsafe_allow_html=True)
    
    # Show history breadcrumbs
    for i, (node_name, option_text) in enumerate(st.session_state.history):
        breadcrumb_class = "breadcrumb-item active" if i == len(st.session_state.history) - 1 else "breadcrumb-item"
        st.markdown(f"<div class='{breadcrumb_class}' style='cursor: pointer;'>{option_text}</div>", 
                   unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("---")