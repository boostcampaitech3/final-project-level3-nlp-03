from pydantic import DecimalMaxDigitsError


def preprocess(data):
  
  data = data.data
  print(data[0].question)

  

  return data