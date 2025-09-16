# 🌟 Professional AV BOQ Generator 🌟

This project is a professional Bill of Quantities (BOQ) generator for Audio-Visual (AV) systems. It employs a logical, system-building approach to create comprehensive and intelligent BOQs based on user requirements. The solution is comprised of a frontend web application, a backend API, and a data ingestion script.

---

## ✨ Features

- **Intelligent BOQ Generation**: Generates a logical and coherent AV system by matching compatible components from a product database.
- **Ecosystem-Based Design**: The system prioritizes products from the same **brand ecosystem** (e.g., Logitech, Cisco, Poly) for guaranteed compatibility and simplified management.
- **Room & Budget-Based Recommendations**: Tailors the generated BOQ based on the specified room size and budget range.
- **Data-Driven Logic**: Utilizes a **SQLite database** of AV products populated by a dedicated data ingestion script.
- **Professional Report Generation**: The frontend presents a professional, printable, and downloadable BOQ report.
- **Robust Data Ingestion**: A dedicated Python script (`ingest_data.py`) processes product data from **CSV and PDF files**, automatically categorizing products, extracting brands, and populating the database.
- **Comprehensive API**: A **FastAPI** backend exposes a set of endpoints for BOQ generation, product management, and health checks.
- **Interactive Frontend**: The web-based user interface (`index.html`) provides a simple way to configure project details and generate the BOQ.

---

## 🏗️ Project Structure

The project is logically divided into three primary components:

- `main.py`: The FastAPI backend containing the core business logic for BOQ generation. This includes classes for managing product ecosystems (`EcosystemManager`), matching products (`IntelligentProductMatcher`), and assembling the final system (`LogicalSystemBuilder`).
- `ingest_data.py`: A Python script responsible for importing and processing product data from various sources (CSV, PDF) into the `products.db` SQLite database.
- `index.html`: The frontend user interface. It's a single HTML file that uses embedded CSS (Tailwind) and JavaScript to provide a dynamic and responsive experience.

---

## 🧠 How It Works

1.  **Data Ingestion**: The `ingest_data.py` script scans a `data` folder for product files. It extracts, cleans, and enriches data by inferring attributes like `brand`, `category`, and `tier` using keyword matching and NLP with the SpaCy library. The processed data is then inserted into `products.db`.
2.  **User Input**: A user configures the project in `index.html`, specifying details like project name, room size, and budget.
3.  **API Request**: The frontend sends a `POST` request with the configuration to the `/api/generate_boq` endpoint.
4.  **Backend Processing**:
    - The backend uses `EcosystemManager` to detect brand preferences from the requirements.
    - `IntelligentProductMatcher` queries the database for compatible products within the chosen ecosystem.
    - `LogicalSystemBuilder` assembles a complete system, including a core VC device, a controller, displays, mounts, and cables, as well as a professional services section.
5.  **BOQ Generation**: The backend compiles all components into a structured BOQ object with pricing and a summary, and sends it back to the frontend.
6.  **Frontend Display**: The API response is used by `index.html` to dynamically render a professional, multi-section report that can be downloaded as a CSV or printed directly.

---

## 🛠️ Setup and Installation

### Prerequisites

- Python 3.8+
- `pip` (Python package installer)

### Installation Steps

1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd <project_folder>
    ```

2.  **Install dependencies**:
    ```bash
    pip install "fastapi[all]" uvicorn pandas pdfplumber spacy
    python -m spacy download en_core_web_sm
    ```

### Running the Application

1.  **Prepare the Data**:
    - Create a folder named `data` in the project root.
    - Place your product data files (CSV or PDF) inside this folder.

2.  **Run the Data Ingestion Script**:
    This will create the `products.db` file and populate it.
    ```bash
    python ingest_data.py
    ```
    A `data_quality_report_<timestamp>.json` will be generated with a data quality report.

3.  **Start the Backend API**:
    The API will be accessible at `http://127.0.0.1:8000`.
    ```bash
    uvicorn main:app --reload
    ```

4.  **Open the Frontend**:
    Open the `index.html` file in your web browser to access the user interface.

---

## 🔗 API Endpoints

The FastAPI backend exposes the following key endpoints:

| Endpoint                 | Method | Description                                                                 |
| ------------------------ | ------ | --------------------------------------------------------------------------- |
| `/`                      | `GET`  | Root endpoint with a welcome message and API version.               |
| `/api/health`            | `GET`  | Health check to ensure the backend and database are operational.   |
| `/api/products`          | `GET`  | Retrieves a list of all products from the database.                 |
| `/api/generate_boq`      | `POST` | The main endpoint for generating a BOQ based on a JSON configuration. |
| `/api/categories`        | `GET`  | Lists all unique product categories.                                |
| `/api/brands`            | `GET`  | Lists all unique product brands.                                    |
| `/api/ecosystems`        | `GET`  | Provides details on the pre-defined product ecosystems.           |
| `/api/validate_config`   | `POST` | Validates a BOQ configuration before generation.                    |

---

## 📝 Data Ingestion Details

The `ingest_data.py` script is a critical part of the workflow. It:
- Creates a database backup before any changes are made.
- Initializes the `products` table in `products.db` with the correct schema.
- Processes **CSV and PDF files** from the `data` folder.
- Extracts and cleans product names, prices, and features.
- Infers metadata like `brand`, `category`, `tier`, and `use_case_tags` using keyword matching and the **SpaCy NLP library**.
- Inserts or updates product records, handling duplicates to keep the database clean.
- Generates a detailed data quality report to summarize the ingestion process.

**Note**: To enable PDF ingestion, the `en_core_web_sm` SpaCy model must be downloaded. The script includes a warning if the model is not found, and PDF processing will be disabled in that case.
