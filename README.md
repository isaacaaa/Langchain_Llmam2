# Langchain_Llmam2
This is impletation of Retrieval-Augmented Generation with Pincone, Llmam2 and Langchain
## Setup
```
Get your Pinecone api key, environment and index name then modify your key, enviroment and index name in line 20, 21, 30 of run.py
```

## Environment
```
pip install -r requirements.txt
```


## Run code
```
python run.py
```

## Question:
```
YOLOv7 outperform which models
```

## Similar texts searched by Pincone with sentence-transformers/all-MiniLM-L6-V2 embeddings:
```
[Document(page_content='From the results we see that if compared with YOLOv4,\nYOLOv7 has 75% less parameters, 36% less computation,\nand brings 1.5% higher AP. If compared with state-of-the-\nart YOLOR-CSP, YOLOv7 has 43% fewer parameters, 15%\nless computation, and 0.4% higher AP. In the performance\nof tiny model, compared with YOLOv4-tiny-31, YOLOv7-\ntiny reduces the number of parameters by 39% and the\namount of computation by 49%, but maintains the same AP.\nOn the cloud GPU model, our model can still have a higher'), Document(page_content='From the results we see that if compared with YOLOv4,\nYOLOv7 has 75% less parameters, 36% less computation,\nand brings 1.5% higher AP. If compared with state-of-the-\nart YOLOR-CSP, YOLOv7 has 43% fewer parameters, 15%\nless computation, and 0.4% higher AP. In the performance\nof tiny model, compared with YOLOv4-tiny-31, YOLOv7-\ntiny reduces the number of parameters by 39% and the\namount of computation by 49%, but maintains the same AP.\nOn the cloud GPU model, our model can still have a higher'), Document(page_content='From the results we see that if compared with YOLOv4,\nYOLOv7 has 75% less parameters, 36% less computation,\nand brings 1.5% higher AP. If compared with state-of-the-\nart YOLOR-CSP, YOLOv7 has 43% fewer parameters, 15%\nless computation, and 0.4% higher AP. In the performance\nof tiny model, compared with YOLOv4-tiny-31, YOLOv7-\ntiny reduces the number of parameters by 39% and the\namount of computation by 49%, but maintains the same AP.\nOn the cloud GPU model, our model can still have a higher'), Document(page_content='From the results we see that if compared with YOLOv4,\nYOLOv7 has 75% less parameters, 36% less computation,\nand brings 1.5% higher AP. If compared with state-of-the-\nart YOLOR-CSP, YOLOv7 has 43% fewer parameters, 15%\nless computation, and 0.4% higher AP. In the performance\nof tiny model, compared with YOLOv4-tiny-31, YOLOv7-\ntiny reduces the number of parameters by 39% and the\namount of computation by 49%, but maintains the same AP.\nOn the cloud GPU model, our model can still have a higher')]
```

## Answer by Llama2-7b:
```
Based on the results provided, YOLOv7 outperforms both YOLOv4 and YOLOR-CSP on the given tasks.{'output_text': ' Based on the results provided, YOLOv7 outperforms both YOLOv4 and YOLOR-CSP on the given tasks.'}
```
