# Congressional Speech Generator

## Background

The foundation of this project involves finetuning both BERT and GPT2 models. Both models are used for analysis, while GPT2 alone is used for text generation. The analysis portion calculates an EMI (Evidence Minus Intuition) score. EMI is a measure developed by Aroyehun et al. (https://www.nature.com/articles/s41562-025-02136-2) using word2vec embeddings to measure how closely a congressional speech aligns with evidence over intuition seed vectors. Scores are calculated by embedding seed vectos for evidence and intuition words into a word2vec embedding space created from the 8 million speeches themselves. The scores are calculated by embedding the speech into the word2vec space, calculating the cosine similarities between the speech vector and the seed vectors, and subtracting the difference. Higher scores means the speech is more likely to be based on evidence and logic, while lower scores means the speech is more likely to be based on intuition and emotion. 

While while EMI is interesting as a concept, the paper's approach of using a word2vec model for text embeddings is outdated. As part of another project I am doing for a class on computational social science, I have taken a deep dive into the EMI paper to validate and build upon its findings. One such direction is to explore how contextual embeddings in BERT and GPT could be used to calculate EMI. 

For this task, we start with finetuning the base models for each of these architectures. Luckily the speech text has already been cleaned and processed by the original paper, and can be found here: https://osf.io/z6utw/files/osfstorage. 

After the process of finetuning the models is complete, we analyze the embedding space using a sparse autoencoder. The sparse autoencoder allows us to identify activations that might correlate with evidence or intuition. It is trained speech activation vectors collected from the hidden layers of both BERT and GPT2 finetuned models. 1000000 speeches are used for training both SAEs. This number was largely choosen by compute and time constraints. By calculating latent vectors for both evidence and intuition, we can then calculate EMI from the latent vector of a speech.

## Application

Text generation is done through zero-shot prefix prompting. We give let the user play with knobs and switches such as temperature, topic selection, party of the speech-speaker, and whether or not they support the issue. The template for the prefix prompt looks like this:


```python
    PROMPT = f"Mr. Speaker, I rise today as a member of the {party} party in {stance} of {topic}"
```

The application will then allow users to calculate BERT, GPT2, and word2vec EMI scores of their newly generated speech. Through my other project, I have found that EMI scores for different embedding spaces can mean different things. Word2Vec primarily measures word occurence, BERT embeddings adds context to the words that appear, and GPT2 embeddings are capable of capturing if a speech is argumentatively structured or not, and able to fold that measure into an EMI score. High, low, and neutral scores in each of these embedding spaces mean different things in terms of what the word2vec, BERT, and GPT2 embeddings are capturing. For example, a high Word2Vec score might indicate that a speech is has a lot of evidence based words, but low GPT and BERT EMI score might reveal that the words are mainly decorative and do not improve the logic or persuasiveness of the speech. A high GPT2 score might indicate that a speech is argumentatively structured, but a low Word2Vec and BERT score might indicate that the speech is not grounded in evidence. 

## How to Run the Application
Application is hosted on Azure, using an Azure container app to deploy the backend API docker image. URL can be found here:

https://zealous-glacier-03d5ea010.7.azurestaticapps.net/

## Repository Structure

The following is the repository structure in order of when it was built. Note that due to compute constraints, some of these scripts are not native to this repo. Some scripts, such as the notebooks in /analysis were run on Google Colab, and others such as the scripts in /models were run on a Quest machine with a dedicated GPUs. Due to file size constraints. Not all ouptut or input data files are present in this github repository. 

### /data

Takes the speeches from the original authors (https://osf.io/z6utw/files/osfstorage) and filters them down based on the criteria in the paper. A new file called raw_data/filtered_speeches.csv is created to be used in subsequent analyses and model finetuning. This notebook was run locally.

### /models

This directory is used to store the script for finetuning BERT and GPT-2 models. Each model is fine-tuned on all ~8 million speeches collected by the previous researchers. The hyperparameters for each model can be found in their respective config.py files. The loss function the was optimized was the standard token-level cross entropy on next-token predictions. Train-eval split was 999% train and 1% eval. 



### /analysis

**generate_speech_exploration.ipynb** prototype speech generation function to test out how to generate speeches

**/activation_scripts** - These scripts were used to collect 1000000 speech activations from the hidden layers of BERT and GPT2. It takes the finetuned models, tokenizes the speeches, and runs the model through the layers to collect the activations. This process takes into account distribution overtime and make sure to get a representative sample of speeches from each decade. The results of these scripts can be found in outputs/activations/bert and outputs/activations/gpt2. gpt2-medium and bert-base-uncased were used for this project. bert_last_1000000.npy, bert_metadata_1000000.json, gpt2_last_1000000.npy, and gpt2_last_1000000.json are created. The whole process was run on Quest (hence the slurm scripts). 

**train_sae.ipynb** - This notebook trains sparse autoencoders on the activations collected from the previous step. Two sparse autoencoders are trained, one for BERT and one for GPT2. The SAEs are trained to reconstruct the activations of the model. The SAE is trained on a subset of the activations, and then evaluated on a separate subset of the activations. This process was done on Google Colab using a GPU runtime, and honestly this process is a little messy because it was my first analysis notebook I wrote for colab, so the structure is not the best. The key outputs here are the model weights for SAEs, which can be found in /outputs/activations/sae. 

**sparse_sae_emi_pipeline.ipynb** - This notebook was primarily used for my other project. It analyzes word2vec EMI scores and creates prototype sparse feature vectors for BERT and GPT embeddings. To do this, we first take the 100k speeches closest to the seed vectors in word2vec embedding spaces. That is 100k speeches each for evidence and intuition seed vectors. We get the activations for each model for all 100k respective speeches, and get the sparse representations of the activations through the SAEs. We take the mean of the sparse representations for all 100k speeches to create a prototype sparse feature vector for evidence and intuition by averaging. Many files are created in this notebook for analysis, but the main ones we care about are bert_evidence_prototype.npy / bert_intuition_prototype.npy and gpt2_evidence_prototype.npy / gpt2_intuition_prototype.npy.


**emi_analysis.ipynb** - This notebook analyzes EMI scores and introduces the concept of mean-centering, which is a method for removing irrelevant information from the embedding vectors. It calculates the EMI scores for all 1000000 speeches where I have the activations for. The files produced here are EMI scores for each speech. This notebook was developed for my other project, and is only here because i thought I could reuse it for creating Evidence and Intuition poles. In this project, I dont need the EMI scores of speeches, I need to be able to produce new EMI scores on new speeches. I decided to write another file to calculate the Evidence and Intuition poles needed to calculate EMI score in the backend server. 

**emi_poles.ipynb** - this notebook calculates the centered EMI poles for each model. It produces bert_evidence_prototype_centered.npy / bert_intuition_prototype_centered.npy and gpt2_evidence_prototype_centered.npy / gpt2_intuition_prototype_centered.npy. These poles are used by the backend to calculate EMI scores for new generated speeches



### /backend

Backend APIs for the application. Docker images with the main app designed to spin up BERT and GPT models on startup to reduce latency of first request. FastAPI is used for the backend API. 

**POST /calculate_emi** Calculate EMI scores for word2vec, BERT, and GPT2. Takes the EMI poles and speech as input. 

- w2v: bool, required (run Word2Vec method)
- bert: bool, required (run BERT method)
- gpt2: bool, required (run GPT-2 method)
- text: string, required (text to score)
- Response: { "w2v_emi": 0.47, "bert_emi": -0.23, "gpt2_emi": 0.12 }

**POST /generate_speech**
- party: string, required
- topic: string, required
- stance: string, optional, default "support" (fills in {stance} of)
- max_new_tokens: int, optional, default 200, range 1–1024
- temperature: float, optional, default 0.9, range >0.0–2.0
- seed: int, optional, default null (set for reproducibility)
- Response: { "speech": "...", "party": "...", "topic": "...", "support": true }

### /frontend

Created using Azure Static Web App service. Built using React. Used the Newsprint design prompt for formatting https://www.designprompts.dev/.

## Deployment 
 
 You can't really deploy locally, the local React app is hooked up to the container app api. I know this is not great form but honeslty it was a little easier than spinning up instructions and control process for the app on local. Here are front and backend build instructions so you can test each component if you want.

### frontend deployment

```bash
cd frontend
npm install
npm run build
npm start
```



### backend deployment

```bash
cd backend
docker build -t emi_api:local -f backend/emi_api/dockerfile backend/emi_api
docker run -d --rm -p 8000:8000 --name emi_api emi_api:local
```

```bash
# test /generate_speech
curl -X POST -G "http://localhost:8000/generate_speech" \
  -d "party=Republican" \
  --data-urlencode "topic=infrastructure investment" \
  -d "stance=support" \
  -d "max_new_tokens=200" \
  -d "temperature=0.9" \
  -d "seed=42" \
  -w "\n"

```
```bash 
# test /calculate_emi
curl -X POST -G "http://localhost:8000/calculate_emi" \
  -d "w2v=false" -d "bert=true" -d "gpt2=true" \
  --data-urlencode "text=The data and statistical analysis clearly show the evidence supports this." \
  -w "\n"
```

## Notes on this project
 
 - The backend docker image is fairly large (around 10 gigs) due to the size of the BERT and GPT-2 models. 
 - Originally seperated each post request into their own images, but hosting with container app services only allowed for one app on US Central. To make that process easier and not pay extra monies for more capability, I just smushed the two backend services into one docker image that runs both on startup. This actually ended up working better as we do need the GPT model to be preloaded for both apps, so it was more efficient
 - Not all the files and notebooks I made for this project work due to the fact that some files were too large to push to github. 