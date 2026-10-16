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
        # ---------------- my_crate (55) ----------------
        {"artist": "Miles Davis", "album": "Kind of Blue", "tags": "jazz modal trumpet calm", "source": "my_crate"},
        {"artist": "John Coltrane", "album": "A Love Supreme", "tags": "jazz spiritual saxophone intense", "source": "my_crate"},
        {"artist": "Marvin Gaye", "album": "What's Going On", "tags": "soul smooth conscious warm", "source": "my_crate"},
        {"artist": "Drake", "album": "Take Care", "tags": "hip-hop r&b moody nocturnal introspective", "source": "my_crate"},
        {"artist": "Stevie Wonder", "album": "Hotter Than July", "tags": "funk soul joyful 80s uplifting", "source": "my_crate"},
        {"artist": "Erykah Badu", "album": "Baduizm", "tags": "neo-soul jazzy sultry smooth", "source": "my_crate"},
        {"artist": "Erykah Badu", "album": "Mama's Gun", "tags": "neo-soul funk introspective groove", "source": "my_crate"},
        {"artist": "Erykah Badu", "album": "Worldwide Underground", "tags": "neo-soul experimental groove hypnotic", "source": "my_crate"},
        {"artist": "Erykah Badu", "album": "New Amerykah Part One", "tags": "neo-soul political futuristic dense", "source": "my_crate"},
        {"artist": "Erykah Badu", "album": "New Amerykah Part Two", "tags": "neo-soul smooth romantic warm", "source": "my_crate"},
        {"artist": "Erykah Badu", "album": "But You Caint Use My Phone", "tags": "r&b mixtape lo-fi late-night", "source": "my_crate"},
        {"artist": "D'Angelo", "album": "Voodoo", "tags": "neo-soul funk raw groove deep", "source": "my_crate"},
        {"artist": "Olivia Dean", "album": "The Art of Loving", "tags": "soul pop warm modern tender", "source": "my_crate"},
        {"artist": "The Weeknd", "album": "House of Balloons", "tags": "alt-r&b dark moody atmospheric", "source": "my_crate"},
        {"artist": "Earl Sweatshirt & MIKE", "album": "Pompeii", "tags": "underground hip-hop abstract lo-fi", "source": "my_crate"},
        {"artist": "Miles Davis", "album": "Sketches of Spain", "tags": "jazz orchestral spanish classical cinematic", "source": "my_crate"},
        {"artist": "Kendrick Lamar", "album": "Section.80", "tags": "hip-hop conscious west-coast jazzy", "source": "my_crate"},
        {"artist": "Kendrick Lamar", "album": "good kid, m.A.A.d city", "tags": "hip-hop cinematic storytelling compton", "source": "my_crate"},
        {"artist": "Kendrick Lamar", "album": "To Pimp a Butterfly", "tags": "hip-hop jazz funk conscious sprawling", "source": "my_crate"},
        {"artist": "Kendrick Lamar", "album": "DAMN.", "tags": "hip-hop sharp minimal direct", "source": "my_crate"},
        {"artist": "Kendrick Lamar", "album": "Mr. Morale & the Big Steppers", "tags": "hip-hop introspective therapy piano", "source": "my_crate"},
        {"artist": "Kendrick Lamar", "album": "GNX", "tags": "hip-hop west-coast energetic hyphy", "source": "my_crate"},
        {"artist": "Frank Ocean", "album": "Channel Orange", "tags": "alt-r&b storytelling lush conceptual", "source": "my_crate"},
        {"artist": "Frank Ocean", "album": "Blonde", "tags": "alt-r&b minimal introspective sparse", "source": "my_crate"},
        {"artist": "Frank Ocean", "album": "Endless", "tags": "alt-r&b visual ambient fragmented", "source": "my_crate"},
        {"artist": "Brent Faiyaz", "album": "Sonder Son", "tags": "alt-r&b moody minimal late-night", "source": "my_crate"},
        {"artist": "Brent Faiyaz", "album": "Fuck the World", "tags": "alt-r&b dark toxic smooth", "source": "my_crate"},
        {"artist": "Brent Faiyaz", "album": "Wasteland", "tags": "alt-r&b cinematic toxic sprawling", "source": "my_crate"},
        {"artist": "Brent Faiyaz", "album": "Larger Than Life", "tags": "alt-r&b smooth confident", "source": "my_crate"},
        {"artist": "Michael Jackson", "album": "Thriller", "tags": "pop funk r&b 80s blockbuster dance", "source": "my_crate"},
        {"artist": "Nirvana", "album": "Nevermind", "tags": "grunge rock raw 90s anthemic", "source": "my_crate"},
        {"artist": "Prince", "album": "Sign o' the Times", "tags": "funk pop rock 80s double-album visionary", "source": "my_crate"},
        {"artist": "SZA", "album": "SOS", "tags": "r&b alt confessional sprawling chart-topping", "source": "my_crate"},
        {"artist": "Solange", "album": "A Seat at the Table", "tags": "r&b art-soul meditative minimal black-pride", "source": "my_crate"},
        {"artist": "J. Cole", "album": "2014 Forest Hills Drive", "tags": "hip-hop conscious storytelling introspective", "source": "my_crate"},
        {"artist": "J. Cole", "album": "For Your Eyez Only", "tags": "hip-hop soulful mellow reflective", "source": "my_crate"},
        {"artist": "J. Cole", "album": "The Fall-Off", "tags": "hip-hop ambitious conceptual", "source": "my_crate"},
        {"artist": "Leon Bridges", "album": "Texas Moon", "tags": "soul psychedelic texas groove", "source": "my_crate"},
        {"artist": "Concrete Boys", "album": "It's Us Vol. 2", "tags": "hip-hop memphis underground energetic", "source": "my_crate"},
        {"artist": "Jill Scott", "album": "Who Is Jill Scott?: Words and Sounds Vol. 1", "tags": "neo-soul r&b spoken-word lush", "source": "my_crate"},
        {"artist": "Knxwledge", "album": "1988", "tags": "hip-hop instrumental lo-fi beats soulful", "source": "my_crate"},
        {"artist": "Thundercat", "album": "Drunk", "tags": "funk jazz fusion playful virtuoso", "source": "my_crate"},
        {"artist": "Madvillain", "album": "Madvillainy", "tags": "hip-hop underground abstract classic villainous", "source": "my_crate"},
        {"artist": "Daniel Caesar", "album": "Case Study 01", "tags": "alt-r&b gospel introspective minimal", "source": "my_crate"},
        {"artist": "Daniel Caesar", "album": "Freudian", "tags": "alt-r&b soulful tender gospel-tinged", "source": "my_crate"},
        {"artist": "Solange", "album": "When I Get Home", "tags": "r&b experimental houston hazy", "source": "my_crate"},
        {"artist": "Michael Jackson", "album": "Off the Wall", "tags": "disco funk pop 70s joyful", "source": "my_crate"},
        {"artist": "Aaliyah", "album": "Age Ain't Nothing but a Number", "tags": "r&b 90s smooth new-jack debut", "source": "my_crate"},
        {"artist": "Isaiah Rashad", "album": "The House Is Burning", "tags": "hip-hop southern introspective hazy", "source": "my_crate"},
        {"artist": "Isaiah Rashad", "album": "ITS been Awful", "tags": "hip-hop tde southern", "source": "my_crate"},
        {"artist": "NxWorries", "album": "Yes Lawd!", "tags": "neo-soul funk hip-hop groove smooth", "source": "my_crate"},
        {"artist": "NxWorries", "album": "Why Lawd?", "tags": "neo-soul funk smooth laid-back", "source": "my_crate"},
        {"artist": "Earl Sweatshirt", "album": "Some Rap Songs", "tags": "hip-hop abstract lo-fi fragmented introspective", "source": "my_crate"},
        {"artist": "Curren$y", "album": "Continuance", "tags": "hip-hop southern smooth laid-back hustle", "source": "my_crate"},
        {"artist": "SZA", "album": "Ctrl", "tags": "alt-r&b confessional vulnerable 2010s", "source": "my_crate"},
        # ---------------- apple_100 (100) ----------------
        {"artist": "Lauryn Hill", "album": "The Miseducation of Lauryn Hill", "tags": "neo-soul hip-hop r&b conscious soulful", "source": "apple_100"},
        {"artist": "The Beatles", "album": "Abbey Road", "tags": "rock pop 60s melodic timeless", "source": "apple_100"},
        {"artist": "Prince & The Revolution", "album": "Purple Rain", "tags": "funk rock pop 80s electric", "source": "apple_100"},
        {"artist": "Stevie Wonder", "album": "Songs in the Key of Life", "tags": "soul funk r&b joyful sprawling", "source": "apple_100"},
        {"artist": "Amy Winehouse", "album": "Back to Black", "tags": "soul jazz retro heartbreak", "source": "apple_100"},
        {"artist": "Beyoncé", "album": "Lemonade", "tags": "r&b pop visual bold personal", "source": "apple_100"},
        {"artist": "Fleetwood Mac", "album": "Rumours", "tags": "rock pop 70s harmony california", "source": "apple_100"},
        {"artist": "Radiohead", "album": "OK Computer", "tags": "alt-rock electronic anxious futuristic", "source": "apple_100"},
        {"artist": "Jay-Z", "album": "The Blueprint", "tags": "hip-hop soulful samples confident", "source": "apple_100"},
        {"artist": "Bob Dylan", "album": "Highway 61 Revisited", "tags": "folk rock 60s poetic electric", "source": "apple_100"},
        {"artist": "Adele", "album": "21", "tags": "pop soul heartbreak powerful vocals", "source": "apple_100"},
        {"artist": "Joni Mitchell", "album": "Blue", "tags": "folk singer-songwriter confessional intimate", "source": "apple_100"},
        {"artist": "Taylor Swift", "album": "1989 (Taylor's Version)", "tags": "pop synth 80s bright", "source": "apple_100"},
        {"artist": "Dr. Dre", "album": "The Chronic", "tags": "hip-hop g-funk west-coast pioneering", "source": "apple_100"},
        {"artist": "The Beach Boys", "album": "Pet Sounds", "tags": "pop rock 60s orchestral harmony", "source": "apple_100"},
        {"artist": "The Beatles", "album": "Revolver", "tags": "rock pop 60s experimental psychedelic", "source": "apple_100"},
        {"artist": "Bruce Springsteen", "album": "Born to Run", "tags": "rock heartland anthemic 70s", "source": "apple_100"},
        {"artist": "Daft Punk", "album": "Discovery", "tags": "electronic house french funky robotic", "source": "apple_100"},
        {"artist": "David Bowie", "album": "The Rise and Fall of Ziggy Stardust and the Spiders from Mars", "tags": "rock glam theatrical 70s", "source": "apple_100"},
        {"artist": "Kanye West", "album": "My Beautiful Dark Twisted Fantasy", "tags": "hip-hop maximalist orchestral ambitious", "source": "apple_100"},
        {"artist": "Led Zeppelin", "album": "Led Zeppelin II", "tags": "rock hard blues heavy", "source": "apple_100"},
        {"artist": "Pink Floyd", "album": "The Dark Side of the Moon", "tags": "rock prog psychedelic atmospheric", "source": "apple_100"},
        {"artist": "A Tribe Called Quest", "album": "The Low End Theory", "tags": "hip-hop jazz conscious smooth", "source": "apple_100"},
        {"artist": "Billie Eilish", "album": "When We All Fall Asleep, Where Do We Go?", "tags": "alt-pop dark whispery minimal", "source": "apple_100"},
        {"artist": "Alanis Morissette", "album": "Jagged Little Pill", "tags": "rock alt 90s raw confessional", "source": "apple_100"},
        {"artist": "The Notorious B.I.G.", "album": "Ready to Die", "tags": "hip-hop east-coast storytelling gritty", "source": "apple_100"},
        {"artist": "Radiohead", "album": "Kid A", "tags": "electronic alt-rock experimental icy", "source": "apple_100"},
        {"artist": "Public Enemy", "album": "It Takes a Nation of Millions to Hold Us Back", "tags": "hip-hop political dense revolutionary", "source": "apple_100"},
        {"artist": "The Clash", "album": "London Calling", "tags": "punk rock rebellious eclectic", "source": "apple_100"},
        {"artist": "Beyoncé", "album": "Beyoncé", "tags": "r&b pop visual bold", "source": "apple_100"},
        {"artist": "Wu-Tang Clan", "album": "Enter the Wu-Tang (36 Chambers)", "tags": "hip-hop raw gritty kung-fu", "source": "apple_100"},
        {"artist": "Carole King", "album": "Tapestry", "tags": "singer-songwriter pop 70s warm piano", "source": "apple_100"},
        {"artist": "Nas", "album": "Illmatic", "tags": "hip-hop east-coast poetic boom-bap", "source": "apple_100"},
        {"artist": "Aretha Franklin", "album": "I Never Loved a Man the Way I Love You", "tags": "soul gospel powerful 60s", "source": "apple_100"},
        {"artist": "Outkast", "album": "Aquemini", "tags": "hip-hop funk southern futuristic", "source": "apple_100"},
        {"artist": "Janet Jackson", "album": "Control", "tags": "pop r&b 80s dance assertive", "source": "apple_100"},
        {"artist": "Talking Heads", "album": "Remain in Light", "tags": "new-wave funk art-pop polyrhythmic", "source": "apple_100"},
        {"artist": "Stevie Wonder", "album": "Innervisions", "tags": "soul funk synth 70s conscious", "source": "apple_100"},
        {"artist": "Björk", "album": "Homogenic", "tags": "art-pop electronic strings icelandic", "source": "apple_100"},
        {"artist": "Bob Marley and the Wailers", "album": "Exodus", "tags": "reggae roots spiritual uplifting", "source": "apple_100"},
        {"artist": "Beastie Boys", "album": "Paul's Boutique", "tags": "hip-hop samples funky playful", "source": "apple_100"},
        {"artist": "U2", "album": "The Joshua Tree", "tags": "rock anthemic 80s spacious", "source": "apple_100"},
        {"artist": "Kate Bush", "album": "Hounds of Love", "tags": "art-pop 80s theatrical dreamy", "source": "apple_100"},
        {"artist": "Guns N' Roses", "album": "Appetite for Destruction", "tags": "rock hard rebellious 80s", "source": "apple_100"},
        {"artist": "The Rolling Stones", "album": "Exile on Main Street", "tags": "rock blues 70s loose", "source": "apple_100"},
        {"artist": "Rihanna", "album": "Anti", "tags": "r&b pop moody late-night", "source": "apple_100"},
        {"artist": "The Cure", "album": "Disintegration", "tags": "alt-rock goth dreamy dark", "source": "apple_100"},
        {"artist": "Oasis", "album": "(What's the Story) Morning Glory?", "tags": "britpop rock 90s anthemic", "source": "apple_100"},
        {"artist": "Arctic Monkeys", "album": "AM", "tags": "indie rock nocturnal groove", "source": "apple_100"},
        {"artist": "The Velvet Underground & Nico", "album": "The Velvet Underground & Nico", "tags": "art-rock 60s droning velvet", "source": "apple_100"},
        {"artist": "Sade", "album": "Love Deluxe", "tags": "soul smooth quiet-storm elegant", "source": "apple_100"},
        {"artist": "2Pac", "album": "All Eyez on Me", "tags": "hip-hop west-coast double-album passionate", "source": "apple_100"},
        {"artist": "The Jimi Hendrix Experience", "album": "Are You Experienced", "tags": "rock psychedelic 60s guitar", "source": "apple_100"},
        {"artist": "De La Soul", "album": "3 Feet High and Rising", "tags": "hip-hop playful samples daisy-age", "source": "apple_100"},
        {"artist": "The Smiths", "album": "The Queen Is Dead", "tags": "indie rock 80s jangle witty", "source": "apple_100"},
        {"artist": "Portishead", "album": "Dummy", "tags": "trip-hop dark smoky 90s", "source": "apple_100"},
        {"artist": "The Strokes", "album": "Is This It", "tags": "indie rock garage 2000s cool", "source": "apple_100"},
        {"artist": "Metallica", "album": "Master of Puppets", "tags": "metal thrash heavy 80s", "source": "apple_100"},
        {"artist": "N.W.A", "album": "Straight Outta Compton", "tags": "hip-hop west-coast rebellious raw", "source": "apple_100"},
        {"artist": "Kraftwerk", "album": "Trans-Europe Express", "tags": "electronic krautrock minimal robotic", "source": "apple_100"},
        {"artist": "Steely Dan", "album": "Aja", "tags": "rock jazz smooth studio", "source": "apple_100"},
        {"artist": "Nine Inch Nails", "album": "The Downward Spiral", "tags": "industrial rock dark abrasive", "source": "apple_100"},
        {"artist": "Missy Elliott", "album": "Supa Dupa Fly", "tags": "hip-hop futuristic playful avant-garde", "source": "apple_100"},
        {"artist": "Bad Bunny", "album": "Un Verano Sin Ti", "tags": "reggaeton latin summer beach", "source": "apple_100"},
        {"artist": "Madonna", "album": "Like a Prayer", "tags": "pop 80s dance gospel", "source": "apple_100"},
        {"artist": "Elton John", "album": "Goodbye Yellow Brick Road", "tags": "rock pop 70s piano double-album", "source": "apple_100"},
        {"artist": "Lana Del Rey", "album": "Norman Fucking Rockwell!", "tags": "alt-pop dreamy california sad", "source": "apple_100"},
        {"artist": "Eminem", "album": "The Marshall Mathers LP", "tags": "hip-hop sharp provocative storytelling", "source": "apple_100"},
        {"artist": "Neil Young", "album": "After the Gold Rush", "tags": "folk rock 70s gentle piano", "source": "apple_100"},
        {"artist": "50 Cent", "album": "Get Rich or Die Tryin'", "tags": "hip-hop gritty 2000s street", "source": "apple_100"},
        {"artist": "Patti Smith", "album": "Horses", "tags": "punk rock poetic 70s raw", "source": "apple_100"},
        {"artist": "Snoop Dogg", "album": "Doggystyle", "tags": "hip-hop g-funk west-coast smooth", "source": "apple_100"},
        {"artist": "Kacey Musgraves", "album": "Golden Hour", "tags": "country pop dreamy warm", "source": "apple_100"},
        {"artist": "Mary J. Blige", "album": "My Life", "tags": "r&b hip-hop-soul 90s confessional", "source": "apple_100"},
        {"artist": "Massive Attack", "album": "Blue Lines", "tags": "trip-hop electronic dub smoky", "source": "apple_100"},
        {"artist": "Nina Simone", "album": "I Put a Spell on You", "tags": "jazz soul piano haunting", "source": "apple_100"},
        {"artist": "Lady Gaga", "album": "The Fame Monster", "tags": "pop dance 2000s theatrical", "source": "apple_100"},
        {"artist": "AC/DC", "album": "Back in Black", "tags": "rock hard 80s riffs", "source": "apple_100"},
        {"artist": "George Michael", "album": "Listen Without Prejudice Vol. 1", "tags": "pop soul 90s introspective", "source": "apple_100"},
        {"artist": "Tyler, the Creator", "album": "Flower Boy", "tags": "hip-hop alt lush introspective", "source": "apple_100"},
        {"artist": "Burial", "album": "Untrue", "tags": "dubstep ambient ghostly nocturnal", "source": "apple_100"},
        {"artist": "Usher", "album": "Confessions", "tags": "r&b pop 2000s smooth confessional", "source": "apple_100"},
        {"artist": "Lorde", "album": "Pure Heroine", "tags": "alt-pop minimal teenage sparse", "source": "apple_100"},
        {"artist": "Rage Against the Machine", "album": "Rage Against the Machine", "tags": "rap-metal political explosive funk", "source": "apple_100"},
        {"artist": "Travis Scott", "album": "Astroworld", "tags": "hip-hop trap psychedelic auto-tune", "source": "apple_100"},
        {"artist": "Eagles", "album": "Hotel California", "tags": "rock 70s california harmony", "source": "apple_100"},
        {"artist": "Robyn", "album": "Body Talk", "tags": "pop electronic dance swedish", "source": "apple_100"},
        {"artist": "Michael Jackson", "album": "Thriller", "tags": "pop funk r&b 80s blockbuster dance", "source": "apple_100"},
        {"artist": "Frank Ocean", "album": "Blonde", "tags": "alt-r&b minimal introspective sparse", "source": "apple_100"},
        {"artist": "Kendrick Lamar", "album": "Good Kid, M.A.A.D City", "tags": "hip-hop west-coast narrative coming-of-age", "source": "apple_100"},
        {"artist": "Nirvana", "album": "Nevermind", "tags": "grunge rock raw 90s anthemic", "source": "apple_100"},
        {"artist": "Marvin Gaye", "album": "What's Going On", "tags": "soul smooth conscious warm classic", "source": "apple_100"},
        {"artist": "Miles Davis", "album": "Kind of Blue", "tags": "jazz modal trumpet calm landmark", "source": "apple_100"},
        {"artist": "Drake", "album": "Take Care", "tags": "hip-hop r&b moody nocturnal introspective", "source": "apple_100"},
        {"artist": "Prince", "album": "Sign o' the Times", "tags": "funk pop rock 80s double-album visionary", "source": "apple_100"},
        {"artist": "John Coltrane", "album": "A Love Supreme", "tags": "jazz spiritual saxophone intense suite", "source": "apple_100"},
        {"artist": "D'Angelo", "album": "Voodoo", "tags": "neo-soul funk raw groove deep", "source": "apple_100"},
        {"artist": "Erykah Badu", "album": "Baduizm", "tags": "neo-soul jazzy sultry smooth debut", "source": "apple_100"},
        {"artist": "SZA", "album": "SOS", "tags": "r&b alt confessional sprawling", "source": "apple_100"},
        {"artist": "Solange", "album": "A Seat at the Table", "tags": "r&b art meditative minimal", "source": "apple_100"},
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
    # one row per album: albums in both pools appear once, preferring my_crate
    out["_key"] = out["artist"].str.lower() + "||" + out["album"].str.lower()
    out["_crate"] = (out["source"] == "my_crate").astype(int)
    out = out.sort_values(["match", "_crate"], ascending=[False, False])
    out = out.drop_duplicates(subset=["_key"], keep="first")
    out = out.drop(columns=["_key", "_crate"]).head(n).reset_index(drop=True)
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
    except Exception:
        return None
    return None

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

    def set_query(q):
        st.session_state.query = q

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
        b1.button("\U0001f3ba Voodoo — D'Angelo", use_container_width=True, on_click=set_query, args=("Voodoo",))
        b2.button("\U0001f3ba Baduizm — Erykah Badu", use_container_width=True, on_click=set_query, args=("Baduizm",))
        b3.button("\U0001f3ba To Pimp a Butterfly", use_container_width=True, on_click=set_query, args=("To Pimp a Butterfly",))
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
