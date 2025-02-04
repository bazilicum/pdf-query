# 📄 PDF Query

This project processes and retrieves information from PDF files using Python. It leverages **Qdrant** as a vector database for similarity searches and employs a **Retrieval-Augmented Generation** (RAG).

The system first chunks and encodes text excerpts from PDFs using a sentence transformer, then performs a similarity search based on the user’s query. The retrieved excerpts are then passed to an LLM, which generates a cited response.

The primary goal is to ensure that the LLM’s answers are bounded to the provided text eliminating the possibility of hallucinations and uncertified information and allowing users to easily trace the sources used in the response. Additionally, if the model cannot formulate a relevant answer from the retrieved text, it will simply state: *I don’t know from the provided text*.

---

## 📁 Project Structure

A simplified view of the relevant files:
```
PDF-info-retrival/
│── Dockerfile               
│── .env                     # Environment variables
│── requirements.txt         
│── README.md                
├── app/  
│    ├── static\              # Contains the socket.io js file for the upload functon 
│    ├── templates\           # Contains the service HTML templates 
│    ├── config.py            # sets the sentece trasformer type and the open ai model 
│    ├── app.py               # Main execution script
│    ├── pdf_processor.py     # Handles PDF reading, text cleaning, and chunking
│    └── qdrant_manager.py    # Managing Qdrant collections and vector upserts
│
└── uploads/                  # Holds the uploaded PDFs
```

## 🚀 Getting Started

### Prerequisites

- **Docker**: Ensure Docker is installed.
- **Docker Compose**: Needed for multi-container or advanced Docker setups.

---

## 🏗️ Building and Running the Project

### 1️⃣ Build the Docker Image

From the `PDF-info-retrival` directory (where the `Dockerfile` resides), run:

```bash
docker build -t pdf-info-retrieval .
```

## 🛠️ Docker Compose Setup

Your docker-compose.yml defines two services:
 1.	**pdf_info_retriver** — The main application container **port 5001** → Exposes the Flask API in the pdf_info_retriver service.
 2.	**qdrant** — Qdrant vector database **port 6333** → Exposes Qdrant’s API in the qdrant service.

#### 📄 The docker-compose.yml File
```
version: '3.8'
services:
  pdf_info_retriver:
    build:
      context: /
      dockerfile: Dockerfile
    container_name: pdf_info_retriver
    volumes:
      - <local project folder>:/usr/local/bin/cde
    env_file:
      - .env
    ports:
      - "5001:5001"  # Expose Flask default port
    command: ["tail", "-f", "/dev/null"]
    depends_on:
      - qdrant  # Ensure Qdrant starts first

  qdrant:
    image: qdrant/qdrant:latest
    container_name: qdrant_service
    ports:
      - "6333:6333"  # Maps Qdrant's default port to the host
    volumes:
      - qdrant_storage:/qdrant/storage  # Persistent storage for Qdrant
    restart: unless-stopped

volumes:
  qdrant_storage:
    driver: local
```

### Key points:
- context: The Docker build context is set to the directory where the Dockerfile lives.
- The pdf_info_retriver service uses a volume binding to mount your host code into the container. Replace the name holder to your local project folder
- The **.env** file is used for your private environment variables.<br>
it will store the **OPEN_AI_ORG_ID** and **OPEN_AI_API_KEY**
it will also store the **FLASK_SECRET_KEY** used to encode your session info
- The **config.py** file is used to set up the type of sentence trasformer and the open ai models. Note that the default sentence trasformer chosed is english only. This is done for optimation and speed. I've commented two other trasformers that are multilngual, and obviously takes up more space and processing time.
There you can also configre the DB location and port so as the flask host and port 

## License
This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.