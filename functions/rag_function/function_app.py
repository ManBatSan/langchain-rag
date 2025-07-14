import azure.functions as func
import json
# from langchain_app.rag_client import answer_question, return_test
import logging


app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

@app.route(route="call_rag")
def call_rag(req: func.HttpRequest) -> func.HttpResponse:
    """
    Azure Function HTTP trigger for RAG.
    Expects JSON: {"question": "..."}
    Returns JSON: {"answer": "..."}
    """
    logging.info('Python HTTP trigger function processed a request.')
    
    try:
        payload = req.get_json()
        question = payload.get("question")
        return func.HttpResponse(
            json.dumps({"answer": question, "test": "hhm"}),
            # json.dumps({"answer": question, "test": return_test()}),
            status_code=200,
            mimetype="application/json"
        )
        # if not question:
        #     return func.HttpResponse(
        #         json.dumps({"error": "Missing 'question'"}),
        #         status_code=400,
        #         mimetype="application/json"
        #     )
        # answer = answer_question(question)
        # return func.HttpResponse(
        #     json.dumps({"answer": answer}),
        #     status_code=200,
        #     mimetype="application/json"
        # )
    except Exception as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json"
        )
