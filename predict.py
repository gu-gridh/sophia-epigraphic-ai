#!/usr/bin/env python3
"""
Inference script for SOPHIA model.
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from sophia.inference import InscriptionReader


def main():
    parser = argparse.ArgumentParser(description='Run SOPHIA inference')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--config', required=True, help='Path to model configuration')
    parser.add_argument('--image', help='Path to inscription image')
    parser.add_argument('--annotation', help='Path to annotation JSON file')
    parser.add_argument('--images_dir', help='Directory of images for batch processing')
    parser.add_argument('--output', help='Output file for results')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for processing')
    
    args = parser.parse_args()
    
    # Load model
    print("Loading SOPHIA model")
    reader = InscriptionReader(args.model_path, args.config)
    print("Model loaded successfully!")
    
    results = []
    
    if args.image:
        # Single image inference
        print(f"Processing image: {args.image}")
        
        result = reader.predict(
            image_path=args.image,
            annotations=args.annotation
        )
        
        results.append({
            'image_path': args.image,
            'annotation_path': args.annotation,
            **result
        })
        
        print("Results:")
        print(f"Transcription: {result['transcription']}")
        print(f"Confidence: {result['confidence']:.3f}")
        if result['predicted_dating']:
            dating = result['predicted_dating']
            print(f"Predicted dating: {dating['min_year']:.0f} - {dating['max_year']:.0f} CE")
        
    elif args.images_dir:
        # Batch processing
        print(f"Processing images from directory: {args.images_dir}")
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(Path(args.images_dir).glob(ext))
        
        print(f"Found {len(image_files)} images")
        
        if len(image_files) == 0:
            print("No images found!")
            return 1
        
        # Prepare annotation paths (if they exist)
        annotation_paths = []
        for image_file in image_files:
            # Try to find corresponding annotation
            image_stem = image_file.stem
            annotation_file = Path(args.images_dir).parent / 'annotations' / f'annotation_{image_stem}.json'
            
            if annotation_file.exists():
                annotation_paths.append(str(annotation_file))
            else:
                annotation_paths.append(None)
        
        # Run batch inference
        batch_results = reader.predict_batch(
            image_paths=[str(p) for p in image_files],
            annotations_paths=annotation_paths,
            batch_size=args.batch_size
        )
        
        # Combine results
        for i, result in enumerate(batch_results):
            results.append({
                'image_path': str(image_files[i]),
                'annotation_path': annotation_paths[i],
                **result
            })
        
        print(f"Processed {len(results)} images")
        
        # Print summary
        successful_transcriptions = sum(1 for r in results if r['transcription'] and r['confidence'] > 0.5)
        print(f"Successful transcriptions (confidence > 0.5): {successful_transcriptions}/{len(results)}")
    
    else:
        print("ERROR: Please provide either --image or --images_dir")
        return 1
    
    # Save results
    if args.output:
        print(f"Saving results to: {args.output}")
        
        # Convert numpy arrays to lists for JSON serialization
        for result in results:
            if 'features' in result and result['features'] is not None:
                result['features'] = result['features'].tolist()
        
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print("Results saved!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
