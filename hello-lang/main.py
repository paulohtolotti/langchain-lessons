from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

load_dotenv()

def basic_pipeline():
    # Chain == pipeline architecture
    information = """
    Stevland Hardaway Morris (/ˈstiːvlənd/ STEEV-lənd; né Judkins; born May 13, 1950), known professionally as Stevie Wonder, is an American singer-songwriter, musician, and record producer. He is widely regarded as one of the most influential musicians of the 20th century, and is credited as a pioneer and influence by musicians across a range of genres that include R&B, pop, soul, gospel, funk, and jazz. A virtual one-man band during much of his peak years, Wonder's use of synthesizers and other electronic musical instruments in the 1970s reshaped the conventions of contemporary R&B. He also helped drive such genres into the album era, crafting his LPs as cohesive and consistent, in addition to socially conscious statements with complex compositions.

    Blind since shortly after his birth, Wonder was a child prodigy who signed with Motown's Tamla label at the age of 11, where he was given the professional name Little Stevie Wonder. As a teenager he established himself as one of Motown's most successful acts, known for his harmonica playing and high-pitched singing. His single "Fingertips" (1963) hit No. 1 on the Billboard Hot 100, when he was 13, making him the youngest solo artist ever to top the chart. Wonder's critical and commercial peak, termed his "classic period" (1972–1976), began with the albums Music of My Mind and Talking Book (1972), which abandoned the Motown sound in favor of a synthesizer- and keyboard-driven one. With Innervisions (1973), Fulfillingness' First Finale (1974), and Songs in the Key of Life (1976), he became the first Black musician to win the Grammy Award for Album of the Year and the only artist to have won the award with three consecutive album releases. During the 1970s, he scored the US number-one singles "Superstition", "You Are the Sunshine of My Life", "You Haven't Done Nothin'", "I Wish" and "Sir Duke".

    In the 1980s, Wonder achieved international cultural presence, with high-profile collaborations (notably with Paul McCartney and Michael Jackson), television appearances, charity work, and political influence, including his 1980 campaign to make Martin Luther King Jr.'s birthday a federal holiday in the United States. Hotter Than July (1980), the soundtrack album The Woman in Red (1984), and In Square Circle (1985) all peaked within the top five of the Billboard 200. "Ebony and Ivory", "I Just Called to Say I Love You", "Part-Time Lover", and "That's What Friends Are For" all reached number one, making him the first act to top the Billboard Hot 100 in three consecutive decades. Wonder returned to the top five with his latest album, A Time to Love (2005), and he has continued to remain active in music and political causes.

    Wonder is one of the best-selling music artists of all time, with sales of more than 100 million records worldwide. He has won 25 Grammy Awards (the most by a male solo artist) and an Academy Award. He has been inducted into the Rhythm and Blues Music Hall of Fame, the Rock and Roll Hall of Fame and the Songwriters Hall of Fame. He ranked in the top 10 on Rolling Stone's lists of the greatest singers and greatest songwriters of all time. In 2009, Wonder was named a United Nations Messenger of Peace, and in 2014, he was honored with the Presidential Medal of Freedom. Believing himself to be of Ghanaian ancestry, he was conferred Ghanaian citizenship in 2024.
    """

    input_template = """
    given the information {information} about a person I want you to create:
    1. Short summary
    2. Two interesting topics
    """

    prompt_template = PromptTemplate(
        input_variables=["information"], template=input_template
    )

    model_gpt = ChatOpenAI(temperature=0, model="gpt-5")
    model_groq = ChatGroq(temperature=0, model="llama-3.1-8b-instant")

    chain = prompt_template | model_groq
    response = chain.invoke(input={"information": information})

    print(response.content)

def main():
    basic_pipeline()

if __name__ == "__main__":
    main()
