# LinkInsights

LinkInsights is a powerful tool that allows users to input links from YouTube videos or websites (or both) and scrape data using LangChain's WebLoader and YouTubeLoader. The scraped data is then utilized to generate summaries, reports, or conduct Q&A using large language models (LLMs). This project is implemented in Streamlit, making it easy to interact with and visualize the output.

## Features

- **Input Links**: Accepts both YouTube video links and website links.
- **Data Scraping**: Uses LangChain's WebLoader for websites and YouTubeLoader for video transcripts.
- **Summarization**: Generate concise summaries of the scraped content.
- **Report Generation**: Create detailed reports based on the input data.
- **Q&A Functionality**: Ask questions about the scraped content and receive insightful answers.
- **LLM Integration**: Utilizes Gemini-pro 1.5 for text processing and response generation.

## How It Works

1. **Input the URL**: Provide either a YouTube video link or a website link.
2. **Data Scraping**: The application scrapes relevant data (transcripts or website content).
3. **LLM Processing**: The scraped content is fed into the Gemini Pro 1.5 LLM via LangChain to generate results.
4. **Outputs**: 
    - **Summarization**: A concise summary of the content is displayed.
    - **Q&A**: Ask questions and receive answers from the content.
    - **Report**: A detailed report is generated based on the content.
  








## Technologies Used

- **Streamlit**: Frontend framework for building the user interface.
- **LangChain**: Framework for chaining LLM prompts and connecting with the LLMs.
- **Gemini Pro 1.5**: The LLM used for generating summaries, reports, and Q&A.
- **BeautifulSoup**: For scraping website content.
- **YouTube Transcript API**: For extracting transcripts from YouTube videos.

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/linkinsights.git
   cd linkinsights
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:

   ```bash
   streamlit run main.py
   ```
4. Create a .env file with  GOOGLE_API_KEY:
   ```bash
   GOOGLE_API_KEY=YOUR_API_KEY
   ```

## Usage

1. Open the Streamlit app on your browser after running the above command.
2. Enter either a YouTube video link or a website URL.
3. Select the type of output you want (Summary, Report, or Q&A).
4. Wait for the results to be generated by the LLM, and view the results on the interface.

