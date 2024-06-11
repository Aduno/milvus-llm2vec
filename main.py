from MilvusDAL import MilvusDAL

documents = [
    "As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    "Definition of summit for English Language Learners. : 1  the highest point of a mountain : the top of a mountain. : 2  the highest level. : 3  a meeting or series of meetings between the leaders of two or more governments.",
    "In 2024, the average temperature in Canada for the winter was 0.5 degrees Celsius. This was the warmest winter on record. The previous record was set in 2016, when the average temperature was 0.3 degrees Celsius. The average temperature for the winter in Canada is usually -5 degrees Celsius.",
    "The Toronto school board has decided to implement a new policy that will require all students to wear uniforms. The policy will go into effect at the beginning of the next school year of 2019. The decision was made after a series of meetings between the Toronto school board members and parents. The policy is intended to create a more inclusive and equitable learning environment for all students.",
]
instruction = (
    "Given a web search query, retrieve relevant passages that answer the query:"
)
queries = [
    [instruction, "how much protein should a female eat"],
    [instruction, "What did the Toronto school board decide to implement?"],
]

dal = MilvusDAL()

# dal.delete_collection("documents")
# dal.create_collection("documents", dimension=2048)


dal.insert(documents, "documents")

print(dal.query(queries, "documents", limit=1))