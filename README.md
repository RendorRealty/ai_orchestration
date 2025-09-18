# AI Orchestration Agent for Construction & Renovation
This project contains a Google ADK-based AI orchestration agent designed to assist with various construction and renovation tasks. The agent leverages a suite of powerful tools to provide comprehensive project support, from planning and procurement to visualization and payment.

The agent is built to run as an MCP (Multi-capability) server, exposing an API that allows users to interact with its capabilities programmatically.

## Features
The agent is equipped with the following tools to streamline construction and renovation workflows:

* **👷 Find Local Trades**: Locates nearby plumbers and electricians using the Google Places API, providing contact details, ratings, and addresses.
* **🛒 Material Price Search**: Searches for current retail prices of construction materials from Canadian vendors using the Groq API.
* **📐 Floorplan Material Estimation**: Analyzes a 2D floorplan image using OCR to estimate the quantity of materials needed, such as lumber, drywall, and insulation.
* **🧊 3D Model Generation**: Converts a 2D floorplan image into a 3D STL model for visualization, and uploads it to a public URL for easy sharing.
* **💡 HVAC & Electrical Drawings**: Generates professional HVAC and electrical layout drawings from a floorplan image by calling an external layout generation API.
* **💳 Solana Payments**: Creates secure Solana blockchain payment links for contractors to easily request payments from clients.

## Setup and Deployment
1. **Project Structure**:
   ```
   /ai_orchestration
    |-- agent.py
    |-- prompt.py
    |-- __init__.py
    |-- tools/
    |   |-- __init__.py
    |   |-- get_3D_model_from_2D_floorplan.py
    |   |-- get_hvac_and_electrical_drawings.py
    |   |-- get_materials_from_floorplan.py
    |   |-- get_plumber_electrician_details.py
    |   |-- search_for_materials.py
    |   |-- generate_solana_payment_link.py
    |-- .env
    |-- requirements.txt
   ```
1. **Create a Virtual Environment**:
   ```
    # Navigate to your project directory
    cd your-project-directory
    
    # Create a virtual environment
    python3 -m venv venv
    
    # Activate the virtual environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
   ```
1. **Install Dependencies**:
   ```
    pip install -r requirements.txt
   ```
1. **Configure Environment Variables**:
   ```
    # Tells the ADK to use non-Vertex AI models (like Cohere via LiteLLM)
    GOOGLE_GENAI_USE_VERTEXAI=FALSE
    
    # API key for Google Places and Geocoding APIs
    GOOGLE_PLACE_KEY="YOUR_GOOGLE_PLACES_API_KEY"
    
    # API key for the Cohere model used by the agent
    COHERE_API_KEY="YOUR_COHERE_API_KEY"
    
    # API key for material price searches
    GROQ_API_KEY="YOUR_GROQ_API_KEY"
    
    # API key for uploading images to imgbb.com (used by the HVAC/Electrical drawing tool)
    IMGBB_API_KEY="YOUR_IMGBB_API_KEY"
   ```
1. **Deploy the MCP Server**:
   ```
    adk api_serve
   ```
## Important Notes

### Local API Dependency

The HVAC & Electrical Drawings tool (*get_hvac_and_electrical_drawings*) depends on an external API that it expects to be running locally at *http://127.0.0.1:8001*.

For this tool to function correctly, you must also run the layout generation service on that address. If you don't have this service, the tool will fail, but the rest of the agent will continue to function.
   
