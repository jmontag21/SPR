import os
import streamlit as st
from openai import OpenAI
from pdf2image import convert_from_path
import pandas as pd
import json
import base64
import time
import re
import tempfile

# Initialize Streamlit session state for data caching
if 'data' not in st.session_state:
    st.session_state.data = None

if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

# Set up the Streamlit app
st.title("Invoice PDF Processor")
st.write("""
Upload your PDF invoices, and this app will extract relevant information and provide a downloadable CSV file.
""")

# Access the API key from Streamlit Secrets
try:
    API_KEY = st.secrets["openai"]["api_key"]
except KeyError:
    st.error("OpenAI API key not found in Streamlit secrets. Please configure your secrets.toml file.")
    st.stop()

# Initialize the OpenAI client
client = OpenAI(api_key=API_KEY)

# File uploader allows multiple files
uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    current_uploaded_file_names = sorted([file.name for file in uploaded_files])

    if st.session_state.data is None or st.session_state.uploaded_file_names != current_uploaded_file_names:
        st.session_state.uploaded_file_names = current_uploaded_file_names
        st.session_state.data = None

    if st.session_state.data is None:
        with st.spinner("Setting up..."):
            with tempfile.TemporaryDirectory() as temp_dir:
                pdf_files = []
                for uploaded_file in uploaded_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    pdf_files.append(uploaded_file.name)

                st.write(f"Found {len(pdf_files)} PDF file(s).")

                if not pdf_files:
                    st.error("No PDF files found. Please upload at least one PDF.")

                # Field Name Mapping
                field_name_mapping = {
                    'INVOICE NO.': 'INVOICE_NO',
                    'DATE': 'INVOICE_DATE',
                    'PO#': 'PO_NUMBER',
                    'MERCHANDISE NET': 'MERCHANDISE_NET',
                    'FREIGHT CHARGE': 'FREIGHT_CHARGE',
                    'SMALL ORDER FEE': 'SMALL_ORDER_FEE',
                    'INVOICE TOTAL': 'INVOICE_TOTAL'
                }

                def normalize_field_names(data_dict):
                    normalized_dict = {}
                    for key, value in data_dict.items():
                        normalized_key = field_name_mapping.get(key.strip().upper(), key.strip().upper())
                        normalized_dict[normalized_key] = value
                    return normalized_dict
                
                def process_pdf(pdf_file, max_retries=3, retry_delay=5):
                    pdf_path = os.path.join(temp_dir, pdf_file)

                    for attempt in range(max_retries):
                        try:
                            # Convert only the first page of the PDF to an image
                            images = convert_from_path(pdf_path, first_page=1, last_page=1, fmt='png', output_folder=temp_dir)
                            image = images[0]

                            temp_image_name = f'temp_{pdf_file}.png'
                            image_path = os.path.join(temp_dir, temp_image_name)
                            image.save(image_path)

                            with open(image_path, "rb") as image_file:
                                base64_image = base64.b64encode(image_file.read()).decode('utf-8')

                            # Prepare the prompt and messages for the single page
                            messages = [
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": (
                                                "Please extract the following fields from the invoice image and provide the data in JSON format:\n"
                                                "- INVOICE_NO\n"
                                                "- INVOICE_DATE\n"
                                                "- PO_NUMBER (look for 'PO#')\n"
                                                "- MERCHANDISE_NET\n"
                                                "- FREIGHT_CHARGE\n"
                                                "- SMALL_ORDER_FEE\n"
                                                "- INVOICE_TOTAL"
                                            )
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{base64_image}"
                                            }
                                        }
                                    ]
                                }
                            ]

                            # Call the OpenAI API with image input
                            response = client.chat.completions.create(
                                model='gpt-4o',
                                messages=messages,
                                max_tokens=1000
                            )

                            response_text = response.choices[0].message.content

                            try:
                                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                                if json_match:
                                    json_str = json_match.group(0)
                                    extracted_data = json.loads(json_str)
                                else:
                                    raise ValueError("No JSON found in the response")

                                extracted_data = normalize_field_names(extracted_data)

                                # Add the PDF file name to the extracted data
                                extracted_data['pdf_file'] = pdf_file

                                # Clean up the temporary image file
                                os.remove(image_path)

                                return extracted_data  # Success, return the data

                            except json.JSONDecodeError:
                                st.warning(f"Failed to parse JSON for file {pdf_file} on attempt {attempt + 1}")
                                st.info(f"Response text: {response_text}")
                                raise ValueError("JSON parsing failed")

                        except Exception as e:
                            st.warning(f"An error occurred while processing {pdf_file} on attempt {attempt + 1}: {e}")
                            if attempt < max_retries - 1:
                                st.info(f"Retrying {pdf_file} in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            else:
                                st.error(f"All retries failed for {pdf_file}")
                                return None  # Return None after all retries have failed

                status_placeholders = [st.empty() for _ in pdf_files]
                progress_bar = st.progress(0)
                total_files = len(pdf_files)
                data = []

                for idx, pdf_file in enumerate(pdf_files):
                    status_placeholders[0].markdown(f"**{pdf_file}**: Processing...")
                    result = process_pdf(pdf_file)

                    if result is not None:
                        data.append(result)
                    else:
                        status_placeholders[0].error("Failed")

                    progress_bar.progress((idx + 1) / total_files)

                st.write("All files processed.")

                if data:
                    df = pd.DataFrame(data)
                    columns_order = ['INVOICE_NO', 'INVOICE_DATE', 'PO_NUMBER', 'MERCHANDISE_NET', 'FREIGHT_CHARGE', 'SMALL_ORDER_FEE', 'INVOICE_TOTAL', 'pdf_file']
                    df = df.reindex(columns=columns_order)
                    csv_buffer = df.to_csv(index=False).encode('utf-8')
                    st.session_state.data = data
                else:
                    st.error("No data was extracted from the uploaded PDFs.")

    else:
        st.info("Using cached data.")

    if st.session_state.data:
        with st.spinner("Preparing CSV..."):
            df = pd.DataFrame(st.session_state.data)
            columns_order = ['INVOICE_NO', 'INVOICE_DATE', 'PO_NUMBER', 'MERCHANDISE_NET', 'FREIGHT_CHARGE', 'SMALL_ORDER_FEE', 'INVOICE_TOTAL', 'pdf_file']
            df = df.reindex(columns=columns_order)
            csv_buffer = df.to_csv(index=False).encode('utf-8')

        st.success("Processing complete!")
        st.download_button(
            label="Download CSV",
            data=csv_buffer,
            file_name='invoice_data.csv',
            mime='text/csv',
        )

        st.dataframe(df)
    else:
        st.error("No data was extracted from the uploaded PDFs.")
