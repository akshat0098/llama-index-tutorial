## Query Data using Query Engine in LlamaIndex
import llama_index
from getpass import getpass
from huggingface_hub import login
from llama_index.readers.web import SimpleWebPageReader
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.core import SummaryIndex 


HF_TOKEN = "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"

login(token=HF_TOKEN)
""" 
llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-v0.1",token=HF_TOKEN)


documents = SimpleWebPageReader(html_to_text=True).load_data(['https://www.nytimes.com/2024/08/16/world/asia/kashmir-election.html'])

index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("why kashmir is special?")
"""
#print(response)


## Rag from Scratch
from llama_index.core.llama_dataset import download_llama_dataset
from llama_index.core import VectorStoreIndex
import tqdm
from llama_index.core.llama_pack import download_llama_pack
from llama_index.core.evaluation import FaithfulnessEvaluator
import nest_asyncio
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


rag_dataset,documents = download_llama_dataset("BlockchainSolanaDataset","./data")

print(rag_dataset.to_pandas()[:5])

llm = HuggingFaceInferenceAPI(model_name="mistralai/Mistral-7B-v0.1",token=HF_TOKEN)

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

index = VectorStoreIndex.from_documents(documents=documents,embed_model=embed_model)

##query_engine 

query_engine = index.as_query_engine(llm=llm)

RagEvaluatorPack = download_llama_pack("RagEvaluator","./rag_evaluator_pack")

rag_evaluator_pack = RagEvaluatorPack(rag_dataset=rag_dataset,query_engine=query_engine,judge_llm=llm)


print(rag_evaluator_pack)

evaluator_model = FaithfulnessEvaluator(llm=llm)


nest_asyncio.apply()

response_vector = query_engine.query("what is solana blockchain")


eval_result = evaluator_model.evaluate_response(response=response_vector)

print(eval_result.response)