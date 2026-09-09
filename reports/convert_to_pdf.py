
"""
Convert research paper markdown files to PDF
"""
import os
import markdown
from weasyprint import HTML, CSS
from weasyprint.text.fonts import FontConfiguration

def combine_markdown_files():
    """Combine all research paper parts into one markdown"""
    files = [
        'reports/research_paper.md',
        'reports/research_paper_extended.md',
        'reports/research_paper_part3.md'
    ]
    
    combined = []
    for filepath in files:
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                combined.append(content)
                combined.append('\n\n---\n\n')  
    
    return '\n'.join(combined)

def markdown_to_html(md_content):
    """Convert markdown to HTML with extensions"""
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'toc']
    )
    
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>Space Weather Forecasting System - Research Paper</title>
        <style>
            @page { 
                size: A4;
                margin: 2.5cm;
                @bottom-center { 
                    content: counter(page);
                } 
            } 
            body { 
                font-family: 'DejaVu Sans', Arial, sans-serif;
                font-size: 11pt;
                line-height: 1.6;
                color: #333;
            } 
            h1 { 
                font-size: 24pt;
                color: #1a1a1a;
                border-bottom: 2px solid #333;
                padding-bottom: 10px;
                margin-top: 30px;
                page-break-before: always;
            } 
            h2 { 
                font-size: 18pt;
                color: #2a2a2a;
                margin-top: 25px;
            } 
            h3 { 
                font-size: 14pt;
                color: #3a3a3a;
                margin-top: 20px;
            } 
            h4 { 
                font-size: 12pt;
                color: #4a4a4a;
                margin-top: 15px;
            } 
            code { 
                background-color: #f4f4f4;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: 'DejaVu Sans Mono', monospace;
                font-size: 9pt;
            } 
            pre { 
                background-color: #f4f4f4;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                font-size: 9pt;
            } 
            pre code { 
                background-color: transparent;
                padding: 0;
            } 
            table { 
                border-collapse: collapse;
                width: 100%;
                margin: 15px 0;
            } 
            th, td { 
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            } 
            th { 
                background-color: #f2f2f2;
                font-weight: bold;
            } 
            blockquote { 
                border-left: 4px solid #ccc;
                margin-left: 0;
                padding-left: 20px;
                color: #666;
            } 
            .page-break { 
                page-break-after: always;
            } 
            strong { 
                font-weight: bold;
                color: #000;
            } 
            em { 
                font-style: italic;
            } 
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    return full_html

def main():
    print("Combining markdown files...")
    md_content = combine_markdown_files()
    
    print("Converting markdown to HTML...")
    html_content = markdown_to_html(md_content)
    
    
    with open('reports/research_paper_combined.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    print("Saved HTML to reports/research_paper_combined.html")
    
    print("Converting HTML to PDF...")
    font_config = FontConfiguration()
    
    HTML(string=html_content).write_pdf(
        'reports/research_paper_full.pdf',
        font_config=font_config
    )
    
    print("✓ PDF created successfully: reports/research_paper_full.pdf")
    
    
    size_mb = os.path.getsize('reports/research_paper_full.pdf') / (1024 * 1024)
    print(f"  File size: {size_mb:.2f} MB")

if __name__ == '__main__':
    main()
