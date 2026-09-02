# from fakta_chat.rag.dummy_rag import DummyRAG
# from fakta_chat.rag.solr_rag import SolrRAG

# the model that should be used in evaluation, chatUI
# MODELS TO USE

GEMMA_4_26B = "gemma-4-26b-a4b-it"

MODEL_MAP = {
    GEMMA_4_26B: "google/gemma-4-26B-A4B-it",
}

# Gemma 4 chat turns use <|turn>ROLE ... <turn|> (replaces Gemma 3's <start_of_turn>/<end_of_turn>).
# START_TURN_MODEL also injects an empty thinking channel to keep thinking mode OFF by default,
# mirroring apply_chat_template(..., enable_thinking=False) since this codebase hand-builds the
# prompt instead of calling apply_chat_template. Verify byte-for-byte against the deployed
# tokenizer's chat_template.jinja before relying on this in production.
START_TURN_USER = {GEMMA_4_26B: "<|turn>user\n"}
START_TURN_MODEL = {
    GEMMA_4_26B: "<|turn>model\n",
}
END_TURN_USER = {GEMMA_4_26B: "<turn|>\n"}
# gemma-4-26B-A4B-it may spontaneously emit a thought channel even with
# thinking mode off. Google recommends priming the model turn with an
# empty, already-closed thought channel to suppress this.
THOUGHT_STUB = {GEMMA_4_26B: "<|channel>thought\n<channel|>"}

# DEFAULT_MODEL also determines the output format of the service.
# if the endpoint the model is served through,
# a wrapper needs to be added to llm_formatting.py to mimic this style.
DEFAULT_MODEL = GEMMA_4_26B

# AGENT PROMPT TEMPLATES
RAG_TEMPLATE = {
    "name": "RAG",
    "model": DEFAULT_MODEL,
    "description": "Brugeren starter en ny forespørgsel, retter opmærksomheden mod et nyt emne inden for samme kategori, eller er ikke tilfreds med de resourcer de fik sidst. Spørgsmålet kræver ny informationssøgning i MitCFU kataloget.",
    "prompt": """
Du modtager et spørgsmål og nogle resourcer. Du forklarer brugeren hvorfor resourcerne er relevante for deres spørgsmål.
Det er ikke sikkert at nogen af resourcerne er relevante for brugerens spørgsmål.
Du overholder følgende regler:
- Du svarer kun hvis du har modtaget resourcer der er relevante for brugerens spørgsmål.
- Du opfinder aldrig resourcer.
- Du skriver aldrig links til websider.
- Du svarer altid på dansk.
- Hvis ikke du har fundet relevante resourcer forklarer du det, og beder dem omformulere spørgsmålet.
- Dit output er kun dit svar.
""",
}

FOLLOW_UP_TEMPLATE = {
    "name": "FOLLOW_UP",
    "model": DEFAULT_MODEL,
    "description": "Brugeren spørger om noget der tydeligt bygger videre på den forrige besked, uden ønske om supplerende eller alternative resourcer. Spørgsmålet er kort, og uden nyt emne. Svaret kan ofte findes i den tidligere kontekst eller i det tidligere svar. ",
    "prompt": """
Du modtageren chathistorik og de sidste relevante resourcer. Du svarer på brugerens spørgsmål ud fra chathistorikken og resourcerne.
- Du opfinder aldrig kilder.
- Du skriver aldrig links til websider.
- Hvis ikke du har fundet relevante resourcer forklarer du det, og beder dem omformulere spørgsmålet.
- Dit output er kun dit svar, ikke kilderne på dit svar.
- Dit svar er kort og præcist.
""",
}

SIMPLE_TEMPLATE = {
    "name": "SIMPLE",
    "model": DEFAULT_MODEL,
    "description": "svarer på simple ting som hej, tak, og forklaring på hvad MitCFU er.",
    "prompt": """
Brugeren har stillet et spørgsmål der ikke handler om specifikke MitCFU kilder, eller sagt hej, tak eller farvel.
Du svarer høftligt og kortfattet brugeren med en afslappet tone.
""",
}

FALLBACK_TEMPLATE = {
    "name": "FALLBACK",
    "model": DEFAULT_MODEL,
    "description": "hvis spørgsmålet falder uden for alle andre agenter hjælper denne her brugeren på rette spor igen",
    "prompt": """
Brugeren spørger om noget der ikke er relevant for MitCFU. Forklar brugeren at du ikke kan besvare deres spørgsmål,
og bed dem om at spørge om noget andet.    
""",
}


def ROUTER_TEMPLATE():
    ALL_TEMPLATES = [
        SIMPLE_TEMPLATE,
        FALLBACK_TEMPLATE,
        RAG_TEMPLATE,
        FOLLOW_UP_TEMPLATE,
    ]
    return {
        "name": "ROUTER",
        "model": DEFAULT_MODEL,
        "descrption": "vælger hvilken agent der skal svare på den seneste besked.",
        "prompt": """
    Brugeren har sendt en besked, og det er din opgave at bedømme hvilken agent der skal håndtere beskeden.
    - Du modtager en beskrivelse af de agenter du har til rådighed.
    - Du modtager også hele jeres chathistorik.
    - Du bruger chathistorikken til at vælge hvilken agent der skal svare på spørgsmålet.
    - Du svarer altid på dansk.

Du starter med at tænke højt over chathistorikken, så du kan forklare dig selv hvad brugerens intention er med den seneste besked.
    Agent beskrivelser: """
        + ". ".join([f"[{TEMP['name']}] : {TEMP['description']}\n" for TEMP in ALL_TEMPLATES])
        + """
        Agent typer: """
        + ", ".join([f"[{TEMP['name']}]" for TEMP in ALL_TEMPLATES])
        + """\n\n
    """
        + """
Dit svar formateres som json sådan her: 
{
"tanker": "dine tanker her",
"agent": "AGENT_NAVN"
}
    """,
    }


# TOOL PROMPTS
# TODO brug query splitting og query decomposition til bedre RAG
REFORMULATE_TEMPLATE = {
    "name": "REFORMULATOR",
    "model": DEFAULT_MODEL,
    "description": "Omformulerer og inddeler brugerens spørgsmål inden der laves RAG på den.",
    "prompt": """
Du modtager en brugers henvendelse, som der skal foretages RAG på. Der søges i en vektordatabase med lærevejledninger, beskrivelser af
film, bøger, værktøjer, teamer og mange andre ting.
Din opgave består af to dele:
- At identificere den eller de søgninger der indgår i brugerens henvendelse.
- At omformulere den eller de identificerede søgninger, så de bliver mere detaljerede og ligner det der ligger i databasen.
- Du tilføjer en beskrivelse af emnet der efterspørges. Kun emnet, ikke typen af materiale, som fx film eller bog eller målgruppen, fx udskoling.
- Dit svar struktureres i json.

Eksempler:

Brugerens input:
"Hvilke bøger og film kan bruges til at undervise i klimaforandringer i udskolingen?"

Output:
{
  "tanker": "Brugeren ønsker ressourcer (bøger og film) relateret til emnet klimaforandringer målrettet undervisning i udskolingen. Jeg deler spørgsmålet op i to søgninger: én for bøger og én for film, og præciserer konteksten med undervisning og målgruppe.",
  "søgninger": [
    "Bøger om der handler om bæredygtighed, miljø og klimaforandringer til udskolingen",
    "Klimaforandringer er menneskeskabte eller naturlige ændringer i jordens klima, der påvirker temperaturer, vejr og økosystemer"
  ]
}

Brugerens input:
"Hvordan kan jeg bruge Minecraft i danskundervisningen?"

Output:
{
  "tanker": "Brugeren nævner Minecraft og ønsker idéer til brug i danskundervisningen. Jeg identificerer det som en forespørgsel på undervisningsforløb eller vejledninger, hvor Minecraft er anvendt i faget dansk.",
  "søgninger": [
    "Undervisningsforløb med Minecraft i danskundervisning",
    "Idéer til brug af Minecraft i undervisning",
    "Minecraft er et kreativt computerspil, hvor spillere bygger og udforsker virtuelle verdener lavet af blokke"
  ]
}
""",
}


# for evaluation, the model that should be used for comparison
# flag -c skal sættes til True
# ComparisonRAG = SolrRAG


# # for comparing several retrieval models
# from fakta_chat.rag.retrievers.meta_solr_retriever import MetaSolrRetriever
# from fakta_chat.rag.retrievers.bm25_retriever import BM25Retriever
# from fakta_chat.rag.retrievers.mistrale5_instruct_retriever import Mistrale5Retriever
# from fakta_chat.rag.retrievers.ensemblers.reciprocal_rerank import ReciprocalEnsembler
# #from nily
# from fakta_chat.rag.retrievers.solr_retriever import SolrRetriever
# from fakta_chat.rag.retrievers.multilinguale5_large_retriever import EmbeddingRetriever

# Compare_Retrievers = [MetaSolrRetriever, BM25Retriever, Mistrale5Retriever, ReciprocalEnsembler, SolrRetriever, EmbeddingRetriever]
