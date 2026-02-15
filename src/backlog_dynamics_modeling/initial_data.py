from io import StringIO
from typing import Final

import pandas as pd
import streamlit as st

BACKLOG_INITIAL_SIZE: Final[int] = 256

SPRINTS_DATA_CSV: Final[str] = """
sprint,v,d
0,42,+15
1,48,+22
2,52,+18
3,45,-5
4,50,+30
5,41,+12
6,54,-10
7,47,+8
8,39,+25
9,53,-15
10,51,+5
11,44,+40
12,56,-20
13,49,+10
14,55,-8
"""

@st.cache_data
def read_sprints_data() -> pd.DataFrame:
    return pd.read_csv(StringIO(SPRINTS_DATA_CSV))

