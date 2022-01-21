from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import train,predict


app = FastAPI()

class StockIn(BaseModel):
    inputs: list
    

#class StockOut(StockIn):
#    forecast: dict




@app.get("/")
async def root():
	return {"message":"Todo correcto"}



@app.post("/predict")#, response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    inputs = payload.inputs

    prediction_list = predict(inputs)

    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = {"inputs": inputs, "forecast": prediction_list}
    return response_object


@app.post("/train")#, response_model=StockOut, status_code=200)
def get_ptrain():
    return train()