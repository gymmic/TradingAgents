import chromadb
from chromadb.config import Settings
from openai import OpenAI

from data_enrichment import get_fundamentals


class FinancialSituationMemory:
    def __init__(self, name, config):
        if config["backend_url"] == "http://localhost:11434/v1":
            self.embedding = "nomic-embed-text"
        else:
            self.embedding = "text-embedding-3-small"
        self.client = OpenAI(base_url=config["backend_url"])
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        try:
            self.situation_collection = self.chroma_client.get_collection(name=name)
        except Exception:  # noqa: BLE001
            self.situation_collection = self.chroma_client.create_collection(name=name)

    def get_embedding(self, text):
        """Get OpenAI embedding for a text"""
        
        response = self.client.embeddings.create(
            model=self.embedding, input=text
        )
        return response.data[0].embedding

    def add_situations(self, situations_and_advice):
        """Add financial situations and their corresponding advice."""
        documents = []
        metadatas = []
        ids = []
        embeddings = []

        offset = self.situation_collection.count()

        for i, (situation, recommendation) in enumerate(situations_and_advice):
            embedding = self.get_embedding(situation)
            doc_id = str(offset + i)
            ticker = recommendation.get("ticker") if isinstance(recommendation, dict) else None
            fundamentals = get_fundamentals(ticker) if ticker else {}

            recommendation_text = (
                recommendation.get("advice") if isinstance(recommendation, dict) else recommendation
            )

            metadata = {
                "recommendation": recommendation_text,
                "ticker": ticker,
                **fundamentals,
            }

            documents.append(situation)
            metadatas.append(metadata)
            ids.append(doc_id)
            embeddings.append(embedding)

        self.situation_collection.add(
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids,
        )

    def get_memories(self, current_situation, n_matches=1):
        """Find matching recommendations using OpenAI embeddings"""
        query_embedding = self.get_embedding(current_situation)

        results = self.situation_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_matches,
            include=["metadatas", "documents", "distances"],
        )

        matched_results = []
        for i in range(len(results["documents"][0])):
            matched_results.append(
                {
                    "matched_situation": results["documents"][0][i],
                    "recommendation": results["metadatas"][0][i]["recommendation"],
                    "similarity_score": 1 - results["distances"][0][i],
                }
            )

        return matched_results


if __name__ == "__main__":
    # Example usage
    matcher = FinancialSituationMemory()

    # Example data
    example_data = [
        (
            "High inflation rate with rising interest rates and declining consumer spending",
            "Consider defensive sectors like consumer staples and utilities. Review fixed-income portfolio duration.",
        ),
        (
            "Tech sector showing high volatility with increasing institutional selling pressure",
            "Reduce exposure to high-growth tech stocks. Look for value opportunities in established tech companies with strong cash flows.",
        ),
        (
            "Strong dollar affecting emerging markets with increasing forex volatility",
            "Hedge currency exposure in international positions. Consider reducing allocation to emerging market debt.",
        ),
        (
            "Market showing signs of sector rotation with rising yields",
            "Rebalance portfolio to maintain target allocations. Consider increasing exposure to sectors benefiting from higher rates.",
        ),
    ]

    # Add the example situations and recommendations
    matcher.add_situations(example_data)

    # Example query
    current_situation = """
    Market showing increased volatility in tech sector, with institutional investors 
    reducing positions and rising interest rates affecting growth stock valuations
    """

    try:
        recommendations = matcher.get_memories(current_situation, n_matches=2)

        for i, rec in enumerate(recommendations, 1):
            print(f"\nMatch {i}:")
            print(f"Similarity Score: {rec['similarity_score']:.2f}")
            print(f"Matched Situation: {rec['matched_situation']}")
            print(f"Recommendation: {rec['recommendation']}")

    except Exception as e:
        print(f"Error during recommendation: {str(e)}")
