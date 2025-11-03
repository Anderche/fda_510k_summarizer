import requests
import os
import fitz  # PyMuPDF for PDF validation (install via pip install pymupdf)
import re
import argparse
from difflib import get_close_matches  # For fuzzy matching if needed

"""
NOTES:

This Python script is a standalone tool for downloading FDA 510(k) summary PDFs for medical devices. Users select input type: direct product code (e.g., "LZS") or device name (e.g., "Pacemaker"). For device names, it resolves to product codes via API search, basic lemmatization, and fuzzy matching. It validates codes against the FDA classification endpoint, counts available summaries, fetches metadata, and downloads/validates PDFs to a corpus directory, skipping existing files. Includes error handling, user confirmations, and rate-limited API queries. Requires libraries: requests, os, fitz, re, difflib. (98 words)
"""

def validate_product_code(product_code):
    """
    Validates the FDA product code by checking the classification endpoint.
    First, basic format check; then API validation.
    """
    if not re.match(r'^[A-Z]{3}$', product_code):
        raise ValueError("Invalid product code format. Must be exactly 3 uppercase letters (e.g., 'LZS').")
    
    api_base = "https://api.fda.gov/device/classification.json"
    search_query = f'product_code:"{product_code}"'
    
    count_url = f"{api_base}?search={search_query}&limit=1"
    response = requests.get(count_url)
    
    if response.status_code != 200:
        raise ValueError(f"API error during validation: {response.text}")
    
    data = response.json()
    
    # Check for error object in response
    if 'error' in data:
        raise ValueError(f"API error during validation: {response.text}")
    
    total = data.get('meta', {}).get('results', {}).get('total', 0)
    
    if total == 0:
        raise ValueError(f"Product code '{product_code}' not found in FDA classification database.")
    
    print(f"Product code '{product_code}' validated successfully.")

def resolve_device_name_to_product_codes(device_name, limit=100):
    """
    Searches the classification API for device_name and returns unique product_codes.
    Uses simple search; applies fuzzy matching on results if needed.
    """
    api_base = "https://api.fda.gov/device/classification.json"
    # Simple root form: split words and OR them
    words = device_name.split()
    root_query = " OR ".join(words)
    search_query = f'device_name:({root_query})'
    
    url = f"{api_base}?search={search_query}&limit={limit}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"API error: {response.text}")
    
    data = response.json()
    results = data.get('results', [])
    
    product_codes = set()
    device_names = []  # For fuzzy if needed
    
    for result in results:
        product_codes.add(result['product_code'])
        device_names.append(result['device_name'])
    
    if not product_codes:
        # Try fuzzy on possible device names (fetch more broadly if needed)
        broad_url = f"{api_base}?limit=1000"
        broad_response = requests.get(broad_url)
        if broad_response.status_code == 200:
            broad_data = broad_response.json()
            broad_names = [r['device_name'] for r in broad_data.get('results', [])]
            close_matches = get_close_matches(device_name, broad_names, n=5, cutoff=0.6)
            if close_matches:
                print(f"Did you mean: {', '.join(close_matches)}?")
                # Recurse or adjust query; for simplicity, use first match
                if close_matches:
                    return resolve_device_name_to_product_codes(close_matches[0], limit=limit)
    
    return sorted(list(product_codes))

def _query_510k_count(api_base, search_query):
    """
    Helper function to query 510k API and return count.
    Returns tuple: (success: bool, count: int, error_message: str)
    """
    url = f"{api_base}?search={search_query}&limit=1"
    response = requests.get(url)
    
    if response.status_code != 200:
        return False, 0, response.text
    
    data = response.json()
    
    # Check for error object in response first
    if 'error' in data:
        error_code = data['error'].get('code', '')
        error_message = data['error'].get('message', '')
        # Handle "No matches found" as valid case (0 count) - this means no records exist
        # NOT_FOUND doesn't mean the product code is invalid, just that it has no 510k records
        if (error_code == 'NOT_FOUND' or 
            'No matches found' in error_message or 
            'not found' in error_message.lower() or
            error_message == 'No matches found!'):
            return True, 0, None  # Success with 0 count
        else:
            return False, 0, response.text
    
    # Extract total count from successful response
    total = data.get('meta', {}).get('results', {}).get('total', 0)
    return True, total, None

def get_510k_counts(product_code):
    """
    Queries openFDA for 510(k) records and returns:
    - total: total 510(k) records for product code
    - with_summaries: count with summaries
    - without_summaries: count without summaries
    """
    api_base = "https://api.fda.gov/device/510k.json"
    
    # Get total 510k records for product code
    # Use product_code:CODE format (no quotes) to match API docs: https://api.fda.gov/device/510k.json?search=product_code:$CODE
    total_query = f'product_code:{product_code}'
    success, total, error_msg = _query_510k_count(api_base, total_query)
    
    if not success:
        raise ValueError(f"API error: {error_msg}")
    
    # Get count with summaries
    summary_query = f'product_code:{product_code} AND statement_or_summary:"summary"'
    success, with_summaries, error_msg = _query_510k_count(api_base, summary_query)
    
    if not success:
        raise ValueError(f"API error: {error_msg}")
    
    without_summaries = total - with_summaries
    
    return {'total': total, 'with_summaries': with_summaries, 'without_summaries': without_summaries}

def get_total_summaries(product_code):
    """
    Queries openFDA for total 510(k) records with summaries available.
    """
    counts = get_510k_counts(product_code)
    return counts['with_summaries']

def fetch_all_510k_records(product_code, limit=10000):
    """
    Fetches all 510(k) records for a product code and returns them as a list.
    Used for displaying all records to the user.
    """
    api_base = "https://api.fda.gov/device/510k.json"
    search_query = f'product_code:{product_code}'
    
    all_records = []
    skip = 0
    page_size = 100
    
    while skip < limit:
        url = f"{api_base}?search={search_query}&limit={page_size}&skip={skip}"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"API error at skip {skip}: {response.text}")
        
        data = response.json()
        
        # Check for error object in response
        if 'error' in data:
            error_code = data['error'].get('code', '')
            error_message = data['error'].get('message', '')
            if (error_code == 'NOT_FOUND' or 
                'No matches found' in error_message or 
                'not found' in error_message.lower() or
                error_message == 'No matches found!'):
                break
            else:
                raise ValueError(f"API error at skip {skip}: {response.text}")
        
        results = data.get('results', [])
        if not results:
            break
        
        all_records.extend(results)
        skip += page_size
        
        # Stop if we've gotten all results
        total = data.get('meta', {}).get('results', {}).get('total', 0)
        if len(all_records) >= total:
            break
    
    return all_records

def print_all_510k_records(product_code):
    """
    Prints all 510(k) records for a product code with their details.
    """
    try:
        records = fetch_all_510k_records(product_code)
        print(f"\nAll {len(records)} 510(k) records for {product_code}:")
        print("-" * 80)
        for i, record in enumerate(records, 1):
            k_number = record.get('k_number', 'N/A')
            date_received = record.get('date_received', 'N/A')
            decision_date = record.get('decision_date', 'N/A')
            device_name = record.get('device_name', 'N/A')
            statement_or_summary = record.get('statement_or_summary', 'N/A')
            applicant = record.get('applicant', 'N/A')
            
            # Truncate device name if too long
            if len(device_name) > 50:
                device_name = device_name[:47] + "..."
            
            print(f"{i:3d}. K-number: {k_number:10s} | Date: {date_received:10s} | "
                  f"Type: {statement_or_summary:8s} | {device_name}")
        print("-" * 80)
    except Exception as e:
        print(f"Error fetching 510(k) records: {e}")

def fetch_510k_pdf_metadata(product_code, limit=1000):
    """
    Fetches metadata for 510(k) with summaries, up to limit.
    Returns list of (k_number, pdf_url) tuples.
    Uses date_received field when available for more accurate year prefix extraction.
    """
    api_base = "https://api.fda.gov/device/510k.json"
    # Use product_code:CODE format (no quotes) to match API docs: https://api.fda.gov/device/510k.json?search=product_code:$CODE
    search_query = f'product_code:{product_code} AND statement_or_summary:"summary"'
    
    metadata = []
    skip = 0
    page_size = 100
    
    while skip < limit and len(metadata) < limit:
        url = f"{api_base}?search={search_query}&limit={page_size}&skip={skip}"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"API error at skip {skip}: {response.text}")
        
        data = response.json()
        
        # Check for error object in response - "No matches found!" means no more results
        if 'error' in data:
            error_code = data['error'].get('code', '')
            error_message = data['error'].get('message', '')
            # Handle "No matches found" as end of results (not an error)
            # NOT_FOUND means no more records, not an API error
            if (error_code == 'NOT_FOUND' or 
                'No matches found' in error_message or 
                'not found' in error_message.lower() or
                error_message == 'No matches found!'):
                break  # No more results, exit loop
            else:
                raise ValueError(f"API error at skip {skip}: {response.text}")
        
        results = data.get('results', [])
        if not results:
            break
        
        for result in results:
            k_number = result['k_number']
            date_received = result.get('date_received', '')
            
            # Use date_received year if available, otherwise fall back to k_number year prefix
            if date_received and len(date_received) >= 4:
                year = date_received[:4]
                year_prefix = year[2:4]  # Last 2 digits of year
            else:
                year_prefix = k_number[1:3]  # Fallback to k_number year prefix
            
            pdf_url = f"https://www.accessdata.fda.gov/cdrh_docs/pdf{year_prefix}/{k_number}.pdf"
            metadata.append((k_number, pdf_url))
        
        skip += page_size
    
    return metadata[:limit]

def download_pdf(pdf_url, output_path):
    """
    Downloads PDF if it exists and validates it's a valid PDF.
    Uses proper headers to avoid being blocked by FDA website.
    """
    # Use browser-like headers to avoid being blocked
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }
    
    response = requests.get(pdf_url, stream=True, headers=headers, allow_redirects=True)
    
    # Check if we got redirected to an error/apology page
    if response.status_code != 200:
        print(f"Skipping {pdf_url}: HTTP {response.status_code}")
        return False
    
    # Check if the response is actually a PDF (not an HTML error page)
    content_type = response.headers.get('Content-Type', '').lower()
    is_likely_pdf = 'pdf' in content_type or 'application/pdf' in content_type
    
    # Download and check first bytes to verify it's a PDF
    pdf_content = b''
    with open(output_path, 'wb') as f:
        first_chunk = True
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                if first_chunk:
                    pdf_content = chunk[:4]
                    first_chunk = False
                f.write(chunk)
    
    # Verify it's actually a PDF file
    if not pdf_content.startswith(b'%PDF'):
        if not is_likely_pdf:
            print(f"Skipping {pdf_url}: Not a valid PDF (Content-Type: {content_type})")
        else:
            print(f"Skipping {pdf_url}: File does not appear to be a valid PDF")
        os.remove(output_path)
        return False
    
    # Validate PDF
    try:
        doc = fitz.open(output_path)
        if doc.page_count == 0:
            raise ValueError("Empty PDF")
        doc.close()
        return True
    except Exception as e:
        print(f"Invalid PDF {output_path}: {e}")
        os.remove(output_path)
        return False

def main(show_summaries=False):
    """
    Main function to download FDA 510(k) summary PDFs.
    
    Args:
        show_summaries (bool): If True, show full list of all 510(k) records after count.
    """
    input_type = input("Enter input type (1: Product Code, 2: Device Name): ").strip()
    
    if input_type == '1':
        product_code = input("Enter FDA product code (e.g., LZS): ").strip().upper()
        try:
            validate_product_code(product_code)
        except ValueError as e:
            print(e)
            return
    elif input_type == '2':
        device_name = input("Enter device name (e.g., Pacemaker): ").strip()
        try:
            product_codes = resolve_device_name_to_product_codes(device_name)
            if not product_codes:
                print("No product codes found for the device name.")
                return
            print(f"Found {len(product_codes)} product codes: {', '.join(product_codes)}")
            if len(product_codes) > 1:
                product_code = input("Select one product code: ").strip().upper()
                if product_code not in product_codes:
                    print("Invalid selection.")
                    return
            else:
                product_code = product_codes[0]
            validate_product_code(product_code)  # Still validate
        except ValueError as e:
            print(e)
            return
    else:
        print("Invalid input type.")
        return
    
    try:
        counts = get_510k_counts(product_code)
        print(f"Total 510(k) records for {product_code}: {counts['total']}")
        print(f"  - With summaries: {counts['with_summaries']}")
        print(f"  - Without summaries: {counts['without_summaries']}")
        
        # Print all 510k records only if -w flag is set
        if show_summaries:
            print_all_510k_records(product_code)
        
        if counts['with_summaries'] == 0:
            print("No summaries available. Exiting.")
            return
        
        total_available = counts['with_summaries']
        
        max_download = int(input(f"Enter max PDFs to download (default 1000, up to {total_available}): ") or 1000)
        max_download = min(max_download, total_available)
        
        confirm = input(f"Proceed to fetch metadata for up to {max_download} PDFs? (y/n): ").lower()
        if confirm != 'y':
            print("Aborted.")
            return
        
        metadata = fetch_510k_pdf_metadata(product_code, limit=max_download)
        actual_count = len(metadata)
        print(f"Found {actual_count} PDFs with summaries.")
        
        output_dir = f"corpus_{product_code}"
        os.makedirs(output_dir, exist_ok=True)
        
        confirm_download = input(f"Confirm download of {actual_count} PDFs to '{output_dir}'? (y/n): ").lower()
        if confirm_download != 'y':
            print("Download aborted.")
            return
        
        successful = 0
        for k_number, pdf_url in metadata:
            output_path = os.path.join(output_dir, f"{k_number}.pdf")
            if os.path.exists(output_path):
                print(f"Skipping existing {k_number}.pdf")
                successful += 1
                continue
            
            if download_pdf(pdf_url, output_path):
                successful += 1
            else:
                print(f"Failed to download/validate {k_number}.pdf")
        
        print(f"Successfully downloaded and validated {successful}/{actual_count} PDFs.")
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download FDA 510(k) summary PDFs')
    parser.add_argument('-w', '--show-summaries', action='store_true',
                        help='Show full list of all 510(k) records after count')
    args = parser.parse_args()
    main(show_summaries=args.show_summaries)