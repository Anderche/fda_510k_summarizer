import unittest
from unittest.mock import patch, MagicMock, mock_open
import get_510k_summary
import os


class TestValidateProductCode(unittest.TestCase):
    
    def test_invalid_format_lowercase(self):
        """Test that lowercase product code raises ValueError"""
        with self.assertRaises(ValueError) as context:
            get_510k_summary.validate_product_code("lzs")
        self.assertIn("Invalid product code format", str(context.exception))
    
    def test_invalid_format_wrong_length(self):
        """Test that wrong length product code raises ValueError"""
        with self.assertRaises(ValueError) as context:
            get_510k_summary.validate_product_code("LZ")
        self.assertIn("Invalid product code format", str(context.exception))
    
    def test_invalid_format_with_numbers(self):
        """Test that product code with numbers raises ValueError"""
        with self.assertRaises(ValueError) as context:
            get_510k_summary.validate_product_code("LZ1")
        self.assertIn("Invalid product code format", str(context.exception))
    
    @patch('get_510k_summary.requests.get')
    def test_api_error_status_code(self, mock_get):
        """Test that API error status code raises ValueError"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            get_510k_summary.validate_product_code("LZS")
        self.assertIn("API error during validation", str(context.exception))
    
    @patch('get_510k_summary.requests.get')
    def test_api_error_in_response(self, mock_get):
        """Test that API error object in response raises ValueError"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'error': {'code': 'ERROR', 'message': 'Some error'}
        }
        mock_response.text = '{"error": {"code": "ERROR"}}'
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            get_510k_summary.validate_product_code("LZS")
        self.assertIn("API error during validation", str(context.exception))
    
    @patch('get_510k_summary.requests.get')
    def test_product_code_not_found(self, mock_get):
        """Test that product code with 0 total raises ValueError"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'meta': {'results': {'total': 0}}
        }
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            get_510k_summary.validate_product_code("XYZ")
        self.assertIn("not found in FDA classification database", str(context.exception))
    
    @patch('get_510k_summary.requests.get')
    def test_valid_product_code(self, mock_get):
        """Test that valid product code passes validation"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'meta': {'results': {'total': 10}}
        }
        mock_get.return_value = mock_response
        
        with patch('builtins.print'):  # Suppress print output
            get_510k_summary.validate_product_code("LZS")
        
        mock_get.assert_called_once()


class TestResolveDeviceNameToProductCodes(unittest.TestCase):
    
    @patch('get_510k_summary.requests.get')
    def test_successful_resolution(self, mock_get):
        """Test successful device name to product codes resolution"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {'product_code': 'LZS', 'device_name': 'Pacemaker'},
                {'product_code': 'MFA', 'device_name': 'Pacemaker'},
                {'product_code': 'LZS', 'device_name': 'Pacemaker'}  # Duplicate
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_510k_summary.resolve_device_name_to_product_codes("pacemaker")
        
        self.assertEqual(sorted(result), ['LZS', 'MFA'])
    
    @patch('get_510k_summary.requests.get')
    def test_api_error(self, mock_get):
        """Test that API error raises ValueError"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            get_510k_summary.resolve_device_name_to_product_codes("pacemaker")
        self.assertIn("API error", str(context.exception))
    
    @patch('get_510k_summary.requests.get')
    @patch('get_510k_summary.get_close_matches')
    def test_no_results_with_fuzzy_matching(self, mock_fuzzy, mock_get):
        """Test fuzzy matching when no direct results found"""
        # First call returns no results
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {'results': []}
        
        # Second call for broad search
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {
            'results': [{'device_name': 'Pacemaker System', 'product_code': 'LZS'}]
        }
        
        # Third call with fuzzy match
        mock_response3 = MagicMock()
        mock_response3.status_code = 200
        mock_response3.json.return_value = {
            'results': [{'device_name': 'Pacemaker System', 'product_code': 'LZS'}]
        }
        
        mock_get.side_effect = [mock_response1, mock_response2, mock_response3]
        mock_fuzzy.return_value = ['Pacemaker System']
        
        result = get_510k_summary.resolve_device_name_to_product_codes("pacemakr")
        
        self.assertEqual(result, ['LZS'])


class TestGet510kCounts(unittest.TestCase):
    
    @patch('get_510k_summary.requests.get')
    def test_successful_count_retrieval(self, mock_get):
        """Test successful retrieval of 510k counts"""
        # Mock response for total count
        mock_response_total = MagicMock()
        mock_response_total.status_code = 200
        mock_response_total.json.return_value = {
            'meta': {'results': {'total': 150}}
        }
        
        # Mock response for summaries count
        mock_response_summaries = MagicMock()
        mock_response_summaries.status_code = 200
        mock_response_summaries.json.return_value = {
            'meta': {'results': {'total': 120}}
        }
        
        mock_get.side_effect = [mock_response_total, mock_response_summaries]
        
        result = get_510k_summary.get_510k_counts("LZS")
        
        self.assertEqual(result['total'], 150)
        self.assertEqual(result['with_summaries'], 120)
        self.assertEqual(result['without_summaries'], 30)
    
    @patch('get_510k_summary.requests.get')
    def test_no_matches_found(self, mock_get):
        """Test handling of NOT_FOUND error"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'error': {'code': 'NOT_FOUND', 'message': 'No matches found!'}
        }
        mock_get.return_value = mock_response
        
        result = get_510k_summary.get_510k_counts("XYZ")
        
        self.assertEqual(result['total'], 0)
        self.assertEqual(result['with_summaries'], 0)
        self.assertEqual(result['without_summaries'], 0)
    
    @patch('get_510k_summary.requests.get')
    def test_summaries_api_error(self, mock_get):
        """Test handling of API error when getting summaries count"""
        mock_response_total = MagicMock()
        mock_response_total.status_code = 200
        mock_response_total.json.return_value = {
            'meta': {'results': {'total': 100}}
        }
        
        mock_response_summaries = MagicMock()
        mock_response_summaries.status_code = 200
        mock_response_summaries.json.return_value = {
            'error': {'code': 'NOT_FOUND', 'message': 'No matches found!'}
        }
        mock_response_summaries.text = '{"error": {"code": "NOT_FOUND"}}'
        
        mock_get.side_effect = [mock_response_total, mock_response_summaries]
        
        result = get_510k_summary.get_510k_counts("LZS")
        
        self.assertEqual(result['total'], 100)
        self.assertEqual(result['with_summaries'], 0)
        self.assertEqual(result['without_summaries'], 100)
    
    @patch('get_510k_summary.requests.get')
    def test_api_status_error(self, mock_get):
        """Test handling of non-200 status code"""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_get.return_value = mock_response
        
        with self.assertRaises(ValueError) as context:
            get_510k_summary.get_510k_counts("LZS")
        self.assertIn("API error", str(context.exception))


class TestGetTotalSummaries(unittest.TestCase):
    
    @patch('get_510k_summary.get_510k_counts')
    def test_get_total_summaries(self, mock_counts):
        """Test that get_total_summaries returns correct count"""
        mock_counts.return_value = {
            'total': 100,
            'with_summaries': 75,
            'without_summaries': 25
        }
        
        result = get_510k_summary.get_total_summaries("LZS")
        
        self.assertEqual(result, 75)
        mock_counts.assert_called_once_with("LZS")


class TestFetch510kPdfMetadata(unittest.TestCase):
    
    @patch('get_510k_summary.requests.get')
    def test_successful_metadata_fetch(self, mock_get):
        """Test successful fetching of PDF metadata"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'results': [
                {'k_number': 'K123456'},
                {'k_number': 'K234567'}
            ]
        }
        mock_get.return_value = mock_response
        
        result = get_510k_summary.fetch_510k_pdf_metadata("LZS", limit=2)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 'K123456')
        self.assertIn('pdf12', result[0][1])  # Year prefix from K123456
    
    @patch('get_510k_summary.requests.get')
    def test_no_matches_found_in_loop(self, mock_get):
        """Test handling of NOT_FOUND error in pagination loop"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'error': {'code': 'NOT_FOUND', 'message': 'No matches found!'}
        }
        mock_response.text = '{"error": {"code": "NOT_FOUND"}}'
        mock_get.return_value = mock_response
        
        result = get_510k_summary.fetch_510k_pdf_metadata("LZS", limit=100)
        
        self.assertEqual(len(result), 0)
    
    @patch('get_510k_summary.requests.get')
    def test_pagination(self, mock_get):
        """Test that pagination works correctly"""
        # First page
        mock_response1 = MagicMock()
        mock_response1.status_code = 200
        mock_response1.json.return_value = {
            'results': [{'k_number': f'K{i:06d}'} for i in range(100)]
        }
        
        # Second page - empty results
        mock_response2 = MagicMock()
        mock_response2.status_code = 200
        mock_response2.json.return_value = {'results': []}
        
        mock_get.side_effect = [mock_response1, mock_response2]
        
        result = get_510k_summary.fetch_510k_pdf_metadata("LZS", limit=150)
        
        self.assertEqual(len(result), 100)


class TestDownloadPdf(unittest.TestCase):
    
    @patch('get_510k_summary.os.remove')
    @patch('get_510k_summary.fitz.open')
    @patch('builtins.open', new_callable=mock_open)
    @patch('get_510k_summary.requests.get')
    def test_successful_download(self, mock_get, mock_file, mock_fitz_open, mock_remove):
        """Test successful PDF download and validation"""
        # Mock HTTP response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'%PDF-1.4', b'content']
        mock_get.return_value = mock_response
        
        # Mock PDF document
        mock_doc = MagicMock()
        mock_doc.page_count = 5
        mock_fitz_open.return_value = mock_doc
        
        result = get_510k_summary.download_pdf("http://example.com/test.pdf", "/tmp/test.pdf")
        
        self.assertTrue(result)
        mock_doc.close.assert_called_once()
        mock_remove.assert_not_called()
    
    @patch('get_510k_summary.requests.get')
    def test_http_error(self, mock_get):
        """Test handling of HTTP error"""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        with patch('builtins.print'):  # Suppress print output
            result = get_510k_summary.download_pdf("http://example.com/test.pdf", "/tmp/test.pdf")
        
        self.assertFalse(result)
    
    @patch('get_510k_summary.os.remove')
    @patch('get_510k_summary.fitz.open')
    @patch('builtins.open', new_callable=mock_open)
    @patch('get_510k_summary.requests.get')
    def test_empty_pdf(self, mock_get, mock_file, mock_fitz_open, mock_remove):
        """Test handling of empty PDF"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'content']
        mock_get.return_value = mock_response
        
        mock_doc = MagicMock()
        mock_doc.page_count = 0
        mock_fitz_open.return_value = mock_doc
        
        with patch('builtins.print'):  # Suppress print output
            result = get_510k_summary.download_pdf("http://example.com/test.pdf", "/tmp/test.pdf")
        
        self.assertFalse(result)
        mock_remove.assert_called_once_with("/tmp/test.pdf")
    
    @patch('get_510k_summary.os.remove')
    @patch('get_510k_summary.fitz.open')
    @patch('builtins.open', new_callable=mock_open)
    @patch('get_510k_summary.requests.get')
    def test_invalid_pdf(self, mock_get, mock_file, mock_fitz_open, mock_remove):
        """Test handling of invalid PDF"""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'not a pdf']
        mock_get.return_value = mock_response
        
        mock_fitz_open.side_effect = Exception("Invalid PDF")
        
        with patch('builtins.print'):  # Suppress print output
            result = get_510k_summary.download_pdf("http://example.com/test.pdf", "/tmp/test.pdf")
        
        self.assertFalse(result)
        mock_remove.assert_called_once_with("/tmp/test.pdf")


if __name__ == '__main__':
    unittest.main()

