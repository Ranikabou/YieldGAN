#!/usr/bin/env python3
"""
Auto Report Generator for Treasury GAN
Run this after each training session to automatically generate comprehensive reports.
"""

import os
import sys
from pathlib import Path

def main():
    """Main automation function."""
    print("🚀 Treasury GAN Auto Report Generator")
    print("=" * 50)
    print("This script will automatically generate a comprehensive report")
    print("with all your GAN results, metrics, and visualizations.")
    print("=" * 50)
    
    # Check if we have the required files
    required_files = [
        'checkpoints/best_model.pth',
        'config/gan_config.yaml'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        print("\nPlease ensure you have trained the model first using:")
        print("   make train")
        return
    
    print("✅ All required files found!")
    print("\n🎯 Starting automatic report generation...")
    
    try:
        # Import and run the report generator
        from generate_simple_report import TreasuryGANSimpleReportGenerator
        
        generator = TreasuryGANSimpleReportGenerator()
        report_path = generator.generate_report()
        
        if report_path:
            print(f"\n🎉 SUCCESS! Report generated automatically!")
            print(f"📊 Comprehensive Report: {report_path}")
            print(f"📁 All files saved in: {generator.report_dir}")
            
            # Show what was created
            print(f"\n📋 Files Generated:")
            print(f"   • Comprehensive Report (PNG)")
            print(f"   • 5 Individual Plot Files")
            print(f"   • Raw Data Files (NPY)")
            print(f"   • Metrics Summary (JSON)")
            print(f"   • Statistics Summary (CSV)")
            print(f"   • Configuration Copy (YAML)")
            print(f"   • Run Summary (TXT)")
            
            print(f"\n💡 Next Steps:")
            print(f"   • View the comprehensive report: {report_path}")
            print(f"   • Check individual plots in: {generator.report_dir}/plots/")
            print(f"   • Analyze metrics in: {generator.report_dir}/metrics/")
            print(f"   • Use data files for further analysis")
            
        else:
            print("❌ Report generation failed.")
            
    except Exception as e:
        print(f"❌ Error during automatic report generation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 