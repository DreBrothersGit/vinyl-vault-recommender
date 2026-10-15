"""
================================================================================
VINYL VAULT — the recommender, with a face.
--------------------------------------------------------------------------------
Content-based recommendations (TF-IDF + cosine similarity) over
artist + album + tags, served as a Streamlit web app. Album artwork comes
from the free iTunes Search API (no key needed).

DEPLOY (Hugging Face Spaces — free):
  1. huggingface.co -> New Space -> SDK: Streamlit, name it "vinyl-vault"
  2. Upload these 3 files: app.py, requirements.txt, README.md
  3. Wait ~2 minutes for the build. Share the URL.

RUN LOCALLY:  pip install -r requirements.txt && streamlit run app.py
================================================================================
"""
import re
from difflib import get_close_matches

import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------------
# 1. DATA
# >>> PASTE YOUR FULL DATASET HERE: replace the list below with your 100+ rows.
# Same 4 columns: artist | album | tags | source ("my_crate" / "apple_100")
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    return pd.DataFrame([
        # ---------------- my_crate ----------------
        {"artist": "Miles Davis",            "album": "Kind of Blue",                  "tags": "jazz modal trumpet calm",                       "source": "my_crate"},
        {"artist": "John Coltrane",          "album": "A Love Supreme",                "tags": "jazz spiritual saxophone intense",              "source": "my_crate"},
        {"artist": "Marvin Gaye",            "album": "What's Going On",               "tags": "soul smooth conscious warm",                    "source": "my_crate"},
        {"artist": "Drake",                  "album": "Take Care",                     "tags": "hip-hop r&b moody nocturnal introspective",     "source": "my_crate"},
        {"artist": "Stevie Wonder",          "album": "Hotter Than July",              "tags": "funk soul joyful 80s uplifting",                "source": "my_crate"},
        {"artist": "Erykah Badu",            "album": "Baduizm",                       "tags": "neo-soul jazzy sultry smooth",                  "source": "my_crate"},
        {"artist": "Erykah Badu",            "album": "Mama's Gun",                    "tags": "neo-soul funk introspective groove",            "source": "my_crate"},
        {"artist": "Erykah Badu",            "album": "Worldwide Underground",         "tags": "neo-soul experimental groove hypnotic",         "source": "my_crate"},
        {"artist": "Erykah Badu",            "album": "New Amerykah Part One",         "tags": "neo-soul political futuristic dense",           "source": "my_crate"},
        {"artist": "Erykah Badu",            "album": "New Amerykah Part Two",         "tags": "neo-soul smooth romantic warm",                 "source": "my_crate"},
        {"artist": "Erykah Badu",            "album": "But You Caint Use My Phone",    "tags": "r&b mixtape lo-fi late-night",                  "source": "my_crate"},
        {"artist": "D'Angelo",               "album": "Voodoo",                        "tags": "neo-soul funk raw groove deep",                 "source": "my_crate"},
        {"artist": "Olivia Dean",            "album": "The Art of Loving",             "tags": "soul pop warm modern tender",                   "source": "my_crate"},
        {"artist": "The Weeknd",             "album": "House of Balloons",             "tags": "alt-r&b dark moody atmospheric",                "source": "my_crate"},
        {"artist": "Earl Sweatshirt & MIKE", "album": "Pompeii",                       "tags": "underground hip-hop abstract lo-fi",            "source": "my_crate"},
        {"artist": "Miles Davis",            "album": "Sketches of Spain",             "tags": "jazz orchestral spanish classical cinematic",   "source": "my_crate"},
        {"artist": "Kendrick Lamar",         "album": "Section.80",                    "tags": "hip-hop conscious west-coast jazzy",            "source": "my_crate"},
        {"artist": "Kendrick Lamar",         "album": "good kid, m.A.A.d city",        "tags": "hip-hop cinematic storytelling compton",        "source": "my_crate"},
        {"artist": "Kendrick Lamar",         "album": "To Pimp a Butterfly",           "tags": "hip-hop jazz funk conscious sprawling",         "source": "my_crate"},
        {"artist": "Kendrick Lamar",         "album": "DAMN.",                         "tags": "hip-hop sharp minimal direct",                  "source": "my_crate"},
        {"artist": "Kendrick Lamar",         "album": "Mr. Morale & the Big Steppers", "tags": "hip-hop introspective therapy piano",           "source": "my_crate"},
        {"artist": "Kendrick Lamar",         "album": "GNX",                           "tags": "hip-hop west-coast energetic hyphy",            "source": "my_crate"},
        # ---------------- apple_100 (sample rows) ----------------
        {"artist": "Lauryn Hill",            "album": "The Miseducation of Lauryn Hill", "tags": "neo-soul hip-hop r&b conscious warm 90s",      "source": "apple_100"},
        {"artist": "Frank Ocean",            "album": "Blonde",                          "tags": "alt-r&b introspective atmospheric minimal",     "source": "apple_100"},
        {"artist": "Prince",                 "album": "Purple Rain",                     "tags": "funk pop rock 80s guitar anthemic",             "source": "apple_100"},
        {"artist": "Stevie Wonder",          "album": "Songs in the Key of Life",        "tags": "soul funk joyful sprawling 70s warm",           "source": "apple_100"},
        {"artist": "Amy Winehouse",          "album": "Back to Black",                   "tags": "soul retro jazzy heartbreak 60s",               "source": "apple_100"},
        {"artist": "Beyoncé",                "album": "Lemonade",                        "tags": "r&b pop political genre-bending bold",          "source": "apple_100"},
        {"artist": "Solange",                "album": "A Seat at the Table",             "tags": "neo-soul minimal political calm warm",          "source": "apple_100"},
        {"artist": "Radiohead",              "album": "OK Computer",                     "tags": "alt-rock experimental anxious atmospheric 90s", "source": "apple_100"},
        {"artist": "Kendrick Lamar",         "album": "good kid, m.A.A.d city",          "tags": "hip-hop cinematic storytelling compton",        "source": "apple_100"},
        {"artist": "D'Angelo",               "album": "Voodoo",                          "tags": "neo-soul funk raw groove deep",                 "source": "apple_100"},
    ])


# ----------------------------------------------------------------------------
# 2. ENGINE — TF-IDF + cosine similarity (from verified v2)
# ----------------------------------------------------------------------------
TAG_WEIGHT = 2  # repeat tags 2x so they outweigh artist/title words


def tokenize(text):
    """Keep hyphenated genres ('neo-soul') and 'r&b' as single tokens."""
    tokens = re.findall(r"[a-z0-9][a-z0-9&'\-]*", text.lower())
    return [t for t in tokens if len(t) > 1]


@st.cache_resource
def build_model():
    """Training (runs once): vectorize every album, precompute similarities."""
    df = load_data()
    df["text"] = (df["artist"] + " " + df["album"] + " "
                  + ((df["tags"] + " ") * TAG_WEIGHT)).str.lower()
    vectorizer = TfidfVectorizer(tokenizer=tokenize, token_pattern=None)
    vectors = vectorizer.fit_transform(df["text"])
    similarity = cosine_similarity(vectors)
    return df, similarity


def _prefer_my_crate(df, indices):
    mine = [i for i in indices if df.at[i, "source"] == "my_crate"]
    return mine[0] if mine else indices[0]


def find_album(df, query):
    """exact -> partial -> fuzzy. Raises ValueError if nothing matches."""
    q = query.strip().lower()
    if not q:
        raise ValueError("Type an album title to get recommendations.")
    titles = df["album"].str.lower()

    exact = df.index[titles == q]
    if len(exact):
        return _prefer_my_crate(df, exact)

    partial = df.index[titles.str.contains(re.escape(q))]
    if len(partial):
        return _prefer_my_crate(df, partial)

    fuzzy = get_close_matches(q, titles.unique(), n=1, cutoff=0.6)
    if fuzzy:
        return _prefer_my_crate(df, df.index[titles == fuzzy[0]])

    suggestions = get_close_matches(q, titles.unique(), n=3, cutoff=0.3)
    raise ValueError(f"No album matching '{query}'. Did you mean: {suggestions or 'n/a'}?")


def recommend(df, similarity, album, n=5, pool="both"):
    """Inference: one row lookup + sort. Returns (anchor_row, results_df)."""
    if pool not in {"my_crate", "apple_100", "both"}:
        raise ValueError('pool must be "my_crate", "apple_100", or "both"')
    idx = find_album(df, album)
    anchor = df.loc[idx]
    is_self = (df["artist"].str.lower() == anchor["artist"].lower()) & \
              (df["album"].str.lower() == anchor["album"].lower())
    mask = ~is_self
    if pool != "both":
        mask &= df["source"] == pool
    out = df.loc[mask, ["artist", "album", "source", "tags"]].copy()
    out["match"] = (similarity[idx, mask.values] * 100).round(1)
    out = out.sort_values("match", ascending=False).head(n).reset_index(drop=True)
    return anchor, out


# ----------------------------------------------------------------------------
# 3. ARTWORK — free iTunes Search API, no key needed
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_artwork(artist, album):
    try:
        resp = requests.get(
            "https://itunes.apple.com/search",
            params={"term": f"{artist} {album}", "media": "music",
                    "entity": "album", "limit": 6},
            timeout=8,
        )
        results = resp.json().get("results", [])
        for r in results:  # prefer an exact title + artist match
            if (r.get("collectionName", "").lower() == album.lower()
                    and artist.lower() in r.get("artistName", "").lower()):
                return r["artworkUrl100"].replace("100x100bb", "600x600bb")
        for r in results:  # then any result by the right artist
            if artist.lower() in r.get("artistName", "").lower():
                return r["artworkUrl100"].replace("100x100bb", "600x600bb")
        if results:
            return results[0]["artworkUrl100"].replace("100x100bb", "600x600bb")
                

# ----------------------------------------------------------------------------
# 4. UI
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Vinyl Vault", page_icon="\U0001f3ba", layout="wide")

    st.markdown("""
    <style>
    .stApp { background-color: #141110; }
    h1, h2, h3, p, div, span, label { color: #f2e9d8; }
    .vault-title { font-size: 2.6rem; letter-spacing: .35rem; margin-bottom: 0; }
    .vault-sub { color: #b8a88f; margin-top: 0; font-style: italic; }
    .no-art {
        aspect-ratio: 1/1; border-radius: 12px;
        background: linear-gradient(135deg, #3a2c1e, #1d1712);
        display: flex; align-items: center; justify-content: center;
        font-size: 3rem;
    }
    .card-album { font-weight: 700; margin: .4rem 0 0; }
    .card-artist { color: #b8a88f; margin: 0 0 .3rem; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="vault-title">\U0001f3ba VINYL VAULT</p>', unsafe_allow_html=True)
    st.markdown('<p class="vault-sub">a recommendation engine trained on my record collection</p>',
                unsafe_allow_html=True)

    if "query" not in st.session_state:
        st.session_state.query = ""

    search_col, pool_col, n_col = st.columns([3, 1.4, 1])
    with search_col:
        st.text_input("Pick an album from the crates", key="query",
                      placeholder='Try "Voodoo"...')
    with pool_col:
        pool = st.selectbox("Pool", ["both", "my_crate", "apple_100"])
    with n_col:
        n = st.slider("Results", 3, 10, 5)

    query = st.session_state.query.strip()

    if not query:
        st.markdown("**Not sure where to start?**")
        b1, b2, b3 = st.columns(3)
        if b1.button("\U0001f3ba Voodoo — D'Angelo", use_container_width=True):
            st.session_state.query = "Voodoo"
            st.rerun()
        if b2.button("\U0001f3ba Baduizm — Erykah Badu", use_container_width=True):
            st.session_state.query = "Baduizm"
            st.rerun()
        if b3.button("\U0001f3ba To Pimp a Butterfly", use_container_width=True):
            st.session_state.query = "To Pimp a Butterfly"
            st.rerun()
        st.stop()

    df, similarity = build_model()
    try:
        anchor, recs = recommend(df, similarity, query, n=n, pool=pool)
    except ValueError as e:
        st.warning(str(e))
        st.stop()

    st.markdown(f"### Because you like *{anchor['album']}* — {anchor['artist']}")
    st.caption(f"pool: {pool} · {len(df)} albums indexed")

    cols = st.columns(min(n, 5))
    for i, (_, row) in enumerate(recs.iterrows()):
        with cols[i % len(cols)]:
            art = get_artwork(row["artist"], row["album"])
            if art:
                st.image(art, use_container_width=True)
            else:
                st.markdown('<div class="no-art">\U0001f3b5</div>', unsafe_allow_html=True)
            st.markdown(f'<p class="card-album">{row["album"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="card-artist">{row["artist"]}</p>', unsafe_allow_html=True)
            st.progress(int(row["match"]))
            st.caption(f'{row["match"]}% match · {row["tags"]}')

    st.markdown("---")
    st.caption("Built with scikit-learn · TF-IDF + cosine similarity · artwork via the iTunes Search API")


if __name__ == "__main__":
    main()
