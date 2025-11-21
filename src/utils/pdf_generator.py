"""
PDF Report Generator
Create beautiful formatted PDF reports of race strategy analysis
"""

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.piecharts import Pie

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import io
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceStrategyPDFReport:
    """Generate comprehensive PDF reports for race strategy analysis"""
    
    def __init__(self, output_path):
        """
        Initialize PDF report generator
        
        Args:
            output_path: Path to save the PDF file or BytesIO buffer
        """
        self.output_path = output_path
        self.doc = SimpleDocTemplate(output_path, pagesize=letter,
                                    rightMargin=0.75*inch, leftMargin=0.75*inch,
                                    topMargin=1*inch, bottomMargin=0.75*inch)
        self.story = []
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1f77b4'),
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#ff7f0e'),
            spaceBefore=20,
            spaceAfter=12,
            fontName='Helvetica-Bold'
        ))
        
        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading3'],
            fontSize=14,
            textColor=colors.HexColor('#2ca02c'),
            spaceBefore=15,
            spaceAfter=10,
            fontName='Helvetica-Bold'
        ))
        
        # Metric style
        self.styles.add(ParagraphStyle(
            name='MetricValue',
            parent=self.styles['Normal'],
            fontSize=18,
            textColor=colors.HexColor('#d62728'),
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
    
    def add_cover_page(self, track: str, race: int, driver: str, timestamp: str = None):
        """
        Add cover page to report
        
        Args:
            track: Track name
            race: Race number
            driver: Driver identifier
            timestamp: Report generation timestamp
        """
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Title
        self.story.append(Spacer(1, 2*inch))
        title = Paragraph("🏁 Race Strategy Analysis Report", self.styles['CustomTitle'])
        self.story.append(title)
        self.story.append(Spacer(1, 0.5*inch))
        
        # Race info
        info_style = ParagraphStyle('InfoStyle', parent=self.styles['Normal'],
                                   fontSize=14, alignment=TA_CENTER)
        
        self.story.append(Paragraph(f"<b>Track:</b> {track.upper()}", info_style))
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph(f"<b>Race:</b> {race}", info_style))
        self.story.append(Spacer(1, 0.2*inch))
        self.story.append(Paragraph(f"<b>Driver:</b> Car #{driver}", info_style))
        self.story.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        subtitle = Paragraph("Toyota GR Cup Series", self.styles['CustomSubtitle'])
        self.story.append(subtitle)
        self.story.append(Spacer(1, 1*inch))
        
        # Timestamp
        timestamp_para = Paragraph(f"Generated: {timestamp}", 
                                  ParagraphStyle('timestamp', parent=self.styles['Normal'],
                                               fontSize=10, alignment=TA_CENTER,
                                               textColor=colors.grey))
        self.story.append(timestamp_para)
        
        self.story.append(PageBreak())
    
    def add_executive_summary(self, summary_data: Dict):
        """
        Add executive summary section
        
        Args:
            summary_data: Dictionary with summary metrics
        """
        self.story.append(Paragraph("Executive Summary", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.3*inch))
        
        # Key metrics table
        data = [
            ['Metric', 'Value', 'Status'],
            ['Total Laps', str(summary_data.get('total_laps', 'N/A')), '✓'],
            ['Best Lap Time', f"{summary_data.get('best_lap', 0):.3f}s", '✓'],
            ['Average Lap Time', f"{summary_data.get('avg_lap', 0):.3f}s", '✓'],
            ['Consistency Score', f"{summary_data.get('consistency', 0):.1f}/100", 
             '✓' if summary_data.get('consistency', 0) > 80 else '⚠'],
            ['Final Tire Life', f"{summary_data.get('final_tire_life', 0)*100:.1f}%",
             '✓' if summary_data.get('final_tire_life', 0) > 0.65 else '⚠'],
        ]
        
        table = Table(data, colWidths=[2.5*inch, 2*inch, 1*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.5*inch))
    
    def add_performance_analysis(self, perf_data: pd.DataFrame):
        """
        Add performance analysis section
        
        Args:
            perf_data: DataFrame with lap-by-lap performance data
        """
        self.story.append(Paragraph("Performance Analysis", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        
        if perf_data.empty:
            self.story.append(Paragraph("No performance data available", self.styles['Normal']))
            return
        
        # Lap time statistics
        self.story.append(Paragraph("Lap Time Statistics", self.styles['SectionHeading']))
        
        stats_data = [
            ['Statistic', 'Value'],
            ['Best Lap', f"{perf_data['LAP_TIME_SECONDS'].min():.3f}s"],
            ['Worst Lap', f"{perf_data['LAP_TIME_SECONDS'].max():.3f}s"],
            ['Mean Lap', f"{perf_data['LAP_TIME_SECONDS'].mean():.3f}s"],
            ['Median Lap', f"{perf_data['LAP_TIME_SECONDS'].median():.3f}s"],
            ['Std Deviation', f"{perf_data['LAP_TIME_SECONDS'].std():.3f}s"],
        ]
        
        stats_table = Table(stats_data, colWidths=[3*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2ca02c')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.story.append(stats_table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_strategy_recommendations(self, recommendations: List[str]):
        """
        Add strategy recommendations section
        
        Args:
            recommendations: List of recommendation strings
        """
        self.story.append(Paragraph("Strategy Recommendations", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        
        for i, rec in enumerate(recommendations, 1):
            bullet = Paragraph(f"{i}. {rec}", self.styles['Normal'])
            self.story.append(bullet)
            self.story.append(Spacer(1, 0.1*inch))
        
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_tire_analysis(self, tire_data: pd.DataFrame):
        """
        Add tire degradation analysis
        
        Args:
            tire_data: DataFrame with tire data
        """
        self.story.append(Paragraph("Tire Degradation Analysis", self.styles['CustomSubtitle']))
        self.story.append(Spacer(1, 0.2*inch))
        
        if tire_data.empty or 'TIRE_LIFE_ESTIMATE' not in tire_data.columns:
            self.story.append(Paragraph("No tire data available", self.styles['Normal']))
            return
        
        # Tire statistics
        tire_stats = [
            ['Metric', 'Value'],
            ['Starting Tire Life', f"{tire_data['TIRE_LIFE_ESTIMATE'].iloc[0]*100:.1f}%"],
            ['Ending Tire Life', f"{tire_data['TIRE_LIFE_ESTIMATE'].iloc[-1]*100:.1f}%"],
            ['Total Degradation', f"{(tire_data['TIRE_LIFE_ESTIMATE'].iloc[0] - tire_data['TIRE_LIFE_ESTIMATE'].iloc[-1])*100:.1f}%"],
            ['Avg Degradation Rate', f"{tire_data['DEGRADATION_RATE'].mean():.4f}s/lap"],
            ['Warning Threshold', '75%'],
            ['Critical Threshold', '65%'],
        ]
        
        tire_table = Table(tire_stats, colWidths=[3*inch, 2*inch])
        tire_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#ff7f0e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.story.append(tire_table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def add_comparison_table(self, comparison_df: pd.DataFrame, title: str):
        """
        Add a comparison table
        
        Args:
            comparison_df: DataFrame to display
            title: Table title
        """
        self.story.append(Paragraph(title, self.styles['SectionHeading']))
        self.story.append(Spacer(1, 0.2*inch))
        
        if comparison_df.empty:
            self.story.append(Paragraph("No data available", self.styles['Normal']))
            return
        
        # Convert DataFrame to table data
        table_data = [comparison_df.columns.tolist()] + comparison_df.values.tolist()
        
        # Format numeric values
        for i in range(1, len(table_data)):
            for j in range(len(table_data[i])):
                if isinstance(table_data[i][j], (int, float)) and not pd.isna(table_data[i][j]):
                    table_data[i][j] = f"{table_data[i][j]:.3f}"
        
        # Create table
        col_widths = [6.5*inch / len(comparison_df.columns)] * len(comparison_df.columns)
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#9467bd')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
        ]))
        
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))
    
    def generate(self):
        """Build and save the PDF report"""
        try:
            self.doc.build(self.story)
            logger.info(f"PDF report generated successfully: {self.output_path}")
            return True
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return False


def generate_race_report(driver_number, track_name: str, race_number: int,
                        current_state: Dict,
                        processed_data: pd.DataFrame,
                        insights: Dict,
                        comparison_data: Optional[pd.DataFrame] = None) -> bytes:
    """
    Generate a complete race strategy report
    
    Args:
        driver_number: Driver car number
        track_name: Track name
        race_number: Race number
        current_state: Current race state dictionary
        processed_data: Processed race data
        insights: AI insights dictionary
        comparison_data: Optional comparison data for multiple drivers
    
    Returns:
        PDF file as bytes
    """
    # Generate in-memory PDF
    pdf_buffer = io.BytesIO()
    report = RaceStrategyPDFReport(pdf_buffer)
    
    # Cover page
    report.add_cover_page(track_name, race_number, f"Car #{driver_number}")
    
    # Executive summary
    summary_data = {
        'total_laps': len(processed_data),
        'best_lap': processed_data['LAP_TIME_SECONDS'].min() if 'LAP_TIME_SECONDS' in processed_data.columns else 0,
        'avg_lap': processed_data['LAP_TIME_SECONDS'].mean() if 'LAP_TIME_SECONDS' in processed_data.columns else 0,
        'consistency': processed_data['CONSISTENCY_SCORE'].mean() if 'CONSISTENCY_SCORE' in processed_data.columns else 0,
        'final_tire_life': processed_data['TIRE_LIFE_ESTIMATE'].iloc[-1] if 'TIRE_LIFE_ESTIMATE' in processed_data.columns and len(processed_data) > 0 else 0,
    }
    report.add_executive_summary(summary_data)
    
    # Performance analysis
    report.add_performance_analysis(processed_data)
    
    # Tire analysis
    report.add_tire_analysis(processed_data)
    
    # Strategy recommendations
    if insights and 'priority_recommendations' in insights:
        report.add_strategy_recommendations(insights['priority_recommendations'][:5])
    
    # Multi-driver comparison if provided
    if comparison_data is not None and not comparison_data.empty:
        report.add_comparison_table(comparison_data)
    
    # Generate PDF
    report.generate()
    
    # Return bytes
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()
