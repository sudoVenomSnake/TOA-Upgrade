from openai import OpenAI
from llama_index import StorageContext, load_index_from_storage
from llama_index.storage.docstore import SimpleDocumentStore
from llama_index.vector_stores import FaissVectorStore
from llama_index.storage.index_store import SimpleIndexStore
from llama_index.query_engine import CitationQueryEngine
import json
import os
from pydantic import BaseModel
from typing import List
from llama_index.program import OpenAIPydanticProgram
import streamlit as st

client = OpenAI(api_key = st.secrets['OPENAI_API_KEY'])

st.set_page_config(layout = 'wide', page_title = 'Tree of Approach')

class Approach(BaseModel):
    title: str
    query_legal_database: str
    query_legal_acts: str

class PetitionApproaches(BaseModel):
    approaches: List[Approach]

st.title("Tree of Approach")
# st.caption("Insurance related legal legislation.")

# with st.expander("Index Selection"):
#     index_choice = st.radio(label = "Please select an index from the options below -", options = os.listdir("Insurance"))

with st.expander("Model Selection"):
    model = st.radio(label = "Please select a large language model -", options = ["gpt-3.5-turbo", "gpt-4-1106-preview"])

# if "loaded" not in st.session_state:
#     with open(f"template.json", "r") as f:
#         template = json.load(f)
#     st.session_state.title = template["title"]
#     st.session_state.index_files = template["index_names"]
#     st.session_state.summaries = template["summaries"]
#     st.session_state.model_choice = template["model_choice"]
#     st.session_state.loaded = True

@st.cache_resource
def initialize():
    storage_context = StorageContext.from_defaults(docstore = SimpleDocumentStore.from_persist_dir(persist_dir = "storage"),
        vector_store = FaissVectorStore.from_persist_dir(persist_dir = "storage"),
        index_store = SimpleIndexStore.from_persist_dir(persist_dir = "storage"))
    index = load_index_from_storage(storage_context = storage_context)
    retriever = index.as_retriever(retriever_mode = 'embedding', similarity_top_k = 20)
    return CitationQueryEngine(retriever)

query_engine = initialize()

@st.cache_data
def generate_petition(input_situation, example_petition):
    response = client.chat.completions.create(model = model,
        messages = [
                    {"role": "system", "content": "You are a helpful assistant who answers questions."},
                    {"role": "user", "content": f"{example_petition}\n\n. Like the example petition above, create a sample petition for the situation {input_situation}. Use Indian laws."},
            ])
    return response.choices[0].message.content

if "petition" not in st.session_state:
    st.session_state.petition = False

if "start_framing_solutions" not in st.session_state:
    st.session_state.start_framing_solutions = False

def petition_submit():
    st.session_state.petition = True
    st.session_state.start_framing_solutions = False

def approach_submit():
    st.session_state.start_framing_solutions = True

example_petition = """[Your Full Name]
[Your Address]
[City, State, Postal Code]
[Email Address]
[Phone Number]
[Date]

To,

The Board of Directors
[Company's Full Name]
[Company's Address]
[City, State, Postal Code]

Subject: Petition challenging the proposed Corporate Restructuring Plan involving the Amalgamation of Subsidiaries

Dear Members of the Board,

I, [Your Full Name], a concerned shareholder of [Company's Full Name], wish to bring your attention to the proposed corporate restructuring plan involving the amalgamation of subsidiaries, announced on [Date of Announcement]. I am writing this petition to highlight the potential adverse impacts of this decision on the company's long-term sustainability, the stakeholders' interests, and to challenge the validity of this decision based on the Company's Act, 2013.

While I understand that the decision was made with the intent to streamline operations and increase efficiency, it is important to evaluate the repercussions carefully. The amalgamation could potentially lead to job losses, reduced competition, and have a detrimental impact on consumers and small businesses.

Under Section 394 of the Companies Act, 2013, I propose that an independent and thorough investigation be conducted to ensure that this restructuring plan serves the best interest of all shareholders, employees, creditors, and other stakeholders of [Company's Full Name].

Furthermore, it is requested that the amalgamation proposal be tabled for discussion and voting at the upcoming general meeting. I believe this will enable a democratic process where shareholders can voice their opinions and concerns, as provided under Section 230(3) of the Companies Act, 2013.

I urge the Board to consider a revised plan or alternative measures that could address the aforementioned issues. To this end, I propose the formation of a committee comprising representatives of shareholders, employees, and independent experts to review the proposed plan.

If there is a failure to address these concerns, I, along with other like-minded shareholders, may be forced to seek legal remedies available to us under Section 245 (Class Action Suits) of the Companies Act, 2013, and other relevant statutes.

I hope you will consider this petition seriously and look forward to engaging in constructive dialogue regarding the proposed restructuring plan.

Yours Sincerely,

[Your Full Name]
[Your Designation - if applicable]
[Signature]

CC:

Registrar of Companies, Ministry of Corporate Affairs, Government of India
Securities and Exchange Board of India
[Company's Full Name] Shareholders' Association
Attachments:

List of supporting shareholders and signatures
Copies of any relevant documentation
"""
placeholder_petition = """[Your Full Name]
[Your Address]
[City, State, Postal Code]
[Email Address]
[Phone Number]
[Date]

To,

The Managing Director
[Insurance Company's Full Name]
[Insurance Company's Address]
[City, State, Postal Code]

Subject: Petition regarding the issuance of an unsolicited insurance policy

Dear Sir/Madam,

I, [Your Full Name], a concerned policyholder, hereby bring to your attention the matter of an insurance policy that was issued to me without my consent or request. I am writing this petition to challenge the validity of this policy issuance and seek redressal in accordance with the relevant laws and regulations in India.

On [Date of Policy Issuance], I received a letter from your company indicating that an insurance policy had been issued in my name. However, I want to emphasize that I did not request or authorize the issuance of this policy. This unauthorized action raises concerns about the ethical and legal practices of your company.

Under Indian laws, specifically, the Insurance Regulatory and Development Authority of India (IRDAI) Act, 1999, insurance companies are mandated to obtain prior consent from policyholders before issuing any policies. Furthermore, the Insurance Ombudsman Rules, 2017 provide a grievance redressal mechanism to address complaints related to insurance policies and services.

I hereby request the following actions to rectify this issue:

1. Immediate cancellation of the unsolicited insurance policy issued in my name.
2. A written confirmation from your company stating the cancellation of the policy and the removal of any financial obligations associated with it.
3. An explanation regarding the circumstances under which this policy was issued without my consent.
4. Assurance that no further unsolicited policies will be issued to me or any other policyholder without their explicit consent.
5. Compensation for any inconvenience, mental anguish, or financial loss caused as a result of this unauthorized policy issuance.

I expect a prompt response addressing these concerns within [Specify a reasonable timeframe, e.g., 15 working days]. Failure to provide a satisfactory resolution may compel me to escalate this matter to the relevant authorities, such as the Insurance Regulatory and Development Authority of India (IRDAI), and pursue legal remedies available under Indian law.

I trust that you will take this matter seriously and initiate immediate action to rectify the situation, ensuring compliance with the applicable laws and regulations governing insurance practices in India.

Thank you for your attention to this matter.

Yours Faithfully,

[Your Full Name]
[Your Policy Number - if applicable]
[Signature]

CC:

Insurance Regulatory and Development Authority of India (IRDAI)
[Insurance Company's Full Name] Grievance Redressal Cell
[Insurance Company's Full Name] Customer Service Department
Attachments:

Copy of the policy issuance letter
Any other relevant documents or correspondence"""

with st.expander("Create/Add Petition", expanded = not st.session_state.petition):
    input_situation = st.text_input(label = 'Create a sample petition for the situation -', value = "some individuals infringing on my company")
    if input_situation:
        res = generate_petition(input_situation, example_petition)
    else:
        res = placeholder_petition
    res_edited = st.text_area("Edit Area", res, height = 500)
    st.button(label = "Submit", on_click = petition_submit)

@st.cache_resource
def get_approaches(petition):
    program = OpenAIPydanticProgram.from_defaults(
            output_cls = PetitionApproaches,
            prompt_template_str = (
                """{petition} \n\n Considering the situation and the corresponding petition, create a potential tree of path that the opposing counsel might follow or take in order to defend against the petition. Do include questions that query a legal database to find relevant statues and precedents. Furthermore, frame questions that are non-trivial and answerable by referencing multiple acts and precedents. The questions to be asked against a legal database has to complex and must refer precedents and acts. The following has to be the structure of the tree: The tree has to have atleast five and maximum 100 branches. the branches will have multiple nodes. Under every node expand upon the potential points to argue upon.  The nodes and branches can overlap. There has to be multiple intersection branches. remember the nodes should be framed as a question or a comlex legal situation. The leaf node should end as a question. The question should be followed up by a legal research question referring to a legal database and legal acts database. The questions should be tailored to indian case laws and acts.
                example approach is : title - Was the decision to amalgamate made in accordance with the Company's Act, 2013?
                query_legal_database - Search for precedent case laws where the Companies Act, 2013 has been interpreted in relation to corporate restructuring plans, specifically amalgamation of subsidiaries.
                query_legal_acts - Cross-reference Section 230 and Section 232 of the Companies Act, 2013 to identify whether all procedural requirements and legal compliances for amalgamation have been followed."""
            ),
            verbose = True,
        )
    return program(petition = petition, description = "Data model for a petition resolution legal approaches")



if st.session_state.petition:
    with st.expander("Approaches", expanded = not st.session_state.start_framing_solutions):
        approaches = get_approaches(res_edited)
        if approaches:
            legal_acts = []
            legal_database = []
            for i in approaches.approaches:
                st.subheader(i.title)
                legal_acts.append(st.text_input(f"**Query Legal Acts -**", value = i.query_legal_acts))
                legal_database.append(st.text_input(f"**Query Legal Database -**", value = i.query_legal_database))
        st.button(label = "Submit Approaches", on_click = approach_submit)

@st.cache_data
def get_legal_acts(q):
    response = client.chat.completions.create(model = model,
        messages = [
                    {"role": "system", "content": "You are a helpful assistant who answers questions."},
                    {"role": "user", "content": f"Based on the petition - \n\"{res_edited}\"\n, a possible approach was created {q}. Give a crisp, concise and legal answer to the approach. Give detailed descriptions of the statures, and a subsequent approach to use them in our case."},
            ])
    return response.choices[0].message.content

@st.cache_data
def get_legal_database(q):
    database_answer = query_engine.query(q +"\nList similar cases. If not exact, list similar cases.")
    # return database_answer.response + " \n\n\n Actual Sources - \n\n\n " + database_answer.source_nodes[0].node.extra_info["case_number"] + " & " +database_answer.source_nodes[1].node.extra_info["case_number"]
    return database_answer.response

if st.session_state.start_framing_solutions:
    col1, col2 = st.columns(2)
    with col1:
        st.header("Query Legal Acts")
    with col2:
        st.header("Query Legal Database")
    for i, j in zip(legal_acts, legal_database):
        with col1:
            with st.expander(i):
                st.write(get_legal_acts(i))
        with col2:
            with st.expander(j):
                st.write(get_legal_database(j))