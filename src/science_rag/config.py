# the model that should be used in evaluation, chatUI
# MODELS TO USE

GEMMA_4_26B = "gemma-4-26b-a4b-it"

# DEFAULT_MODEL also determines the output format of the service.
# if the endpoint the model is served through,
# a new client/model mapping needs to be added in agent_streaming_generator.py to support it.
DEFAULT_MODEL = GEMMA_4_26B

# AGENT PROMPT TEMPLATES
RAG_TEMPLATE = {
    "name": "RAG",
    "description": "Brugeren starter en ny forespørgsel, retter opmærksomheden mod et nyt emne inden for samme kategori, eller er ikke tilfreds med de resourcer de fik sidst. Spørgsmålet kræver ny informationssøgning i ScienceRAG kataloget.",
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
    "description": "svarer på simple ting som hej, tak, og forklaring på hvad ScienceRAG er.",
    "prompt": """
Brugeren har stillet et spørgsmål der ikke handler om specifikke ScienceRAG kilder, eller sagt hej, tak eller farvel.
Du svarer høftligt og kortfattet brugeren med en afslappet tone.
""",
}

FALLBACK_TEMPLATE = {
    "name": "FALLBACK",
    "description": "hvis spørgsmålet falder uden for alle andre agenter hjælper denne her brugeren på rette spor igen",
    "prompt": """
Brugeren spørger om noget der ikke er relevant for ScienceRAG. Forklar brugeren at du ikke kan besvare deres spørgsmål,
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
