
Professional AV BOQ Generator
This project is a professional Bill of Quantities (BOQ) generator for Audio-Visual (AV) systems. It uses a logical system-building approach to create comprehensive and intelligent BOQs based on user requirements. The system is comprised of a frontend web application, a backend API, and a data ingestion script.

Table of Contents
Features

Project Structure

How It Works

Setup and Installation

API Endpoints

Data Ingestion

Features
Intelligent BOQ Generation: Generates a logical and coherent AV system (BOQ) by matching compatible components from a product database.

Ecosystem-Based Design: The system prioritizes products from the same brand ecosystem (e.g., Logitech, Cisco) for guaranteed compatibility and simplified management.

Room and Budget-Based Recommendations: Tailors the generated BOQ based on room size and specified budget range.

Data-Driven Logic: Utilizes a SQLite database of AV products populated by the data ingestion script.

Professional Report Generation: The frontend presents a professional, printable, and downloadable BOQ report.

Robust Data Ingestion: A dedicated Python script (ingest_data.py) processes product data from CSV and PDF files, automatically categorizing products, extracting brands, and populating the database.

Comprehensive API: A FastAPI backend exposes a set of endpoints for BOQ generation, product management, and health checks.

Interactive Frontend: The web-based user interface (index.html) provides a simple way to configure project details and generate the BOQ.

Project Structure
The project consists of three main components:

main.py: The FastAPI backend that contains the core business logic for generating the BOQ. It includes classes for managing product ecosystems, matching products, and building the final system.

ingest_data.py: A Python script responsible for importing product data from various sources (CSV, PDF) into the SQLite database.

index.html: The frontend user interface for interacting with the BOQ generator. It's a single HTML file with embedded CSS (Tailwind CSS) and JavaScript.

How It Works
Data Ingestion: The ingest_data.py script scans a designated data folder for CSV and PDF files. It extracts product information, cleans the data, and enriches it by inferring attributes like brand, category, and tier using keyword matching and NLP. This data is then inserted into the products.db SQLite database.

User Input: The user configures the project via the index.html frontend, providing details such as project name, client name, room size, budget, and any specific brand requirements.

API Request: The frontend sends a POST request to the /api/generate_boq endpoint with the user's configuration.

Backend Processing:

The main.py backend receives the request.

An EcosystemManager component intelligently detects any brand preferences from the user's requirements.

An IntelligentProductMatcher queries the products.db to find suitable and compatible products.

A LogicalSystemBuilder assembles a complete, logical system by selecting a core video conferencing device, a compatible controller, and other necessary components like displays, mounts, and cables.

It also generates a professional services section with installation costs and adds intelligent recommendations.

BOQ Generation: The backend compiles all selected components into a structured BOQ object, including pricing, a summary, and metadata.

Frontend Display: The API returns the BOQ data to the frontend, which dynamically renders a professional, multi-section report in the browser. Users can then download the BOQ as a CSV or print it directly.

Setup and Installation
Prerequisites
Python 3.8+

pip (Python package installer)

Installation Steps
Clone the repository:

Bash

git clone <repository_url>
cd <project_folder>
Install dependencies:

Bash

pip install -r requirements.txt
(Note: A requirements.txt file is not provided, but the necessary packages are fastapi, uvicorn, pandas, sqlite3, pdfplumber, spacy, and typing)

To set up the environment, run:

Bash

pip install "fastapi[all]" uvicorn pandas pdfplumber spacy
python -m spacy download en_core_web_sm
Running the Application
Prepare the Data:

Create a folder named data in the project directory.

Place your AV product data files (CSV or PDF) inside this folder. The ingest_data.py script is designed to process these files.

Run the Data Ingestion Script:
This script will create the products.db file and populate it with your product data.

Bash

python ingest_data.py
A data_quality_report_<timestamp>.json will also be generated, providing insights into the ingested data.

Start the Backend API:
Run the main.py application using uvicorn. The API will be accessible at http://127.0.0.1:8000.

Bash

uvicorn main:app --reload
Open the Frontend:
Open the index.html file in your web browser to access the user interface.

API Endpoints
The FastAPI backend exposes the following key endpoints:

Endpoint	Method	Description
/	GET	Root endpoint with a welcome message and API version.
/api/health	GET	Health check to ensure the backend and database are operational.
/api/products	GET	Retrieves a list of all products from the database.
/api/generate_boq	POST	The main endpoint for generating a BOQ based on a JSON configuration.
/api/categories	GET	Lists all unique product categories.
/api/brands	GET	Lists all unique product brands.
/api/ecosystems	GET	Provides details on the pre-defined product ecosystems.
/api/validate_config	POST	Validates a BOQ configuration before generation.

Export to Sheets
Data Ingestion
The ingest_data.py script is a critical part of the workflow. It:

Creates a database backup before making any changes.

Initializes the products table in products.db with the correct schema.

Processes CSV and PDF files from the data folder.

Extracts and cleans product names, prices, and features.

Infers metadata such as brand, category, tier, and use-case-tags using a combination of keyword matching and the SpaCy NLP library.

Inserts or updates product records in the database, handling duplicates.

Generates a detailed data quality report to summarize the ingestion process.

Note: To use PDF ingestion, the en_core_web_sm SpaCy model must be downloaded. The script includes a warning if the model is not found, in which case PDF processing will be disabled.
