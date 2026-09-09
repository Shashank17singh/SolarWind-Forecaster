import argparse
import json

from model_registry import register_model


def main():
    parser = argparse.ArgumentParser(
        description="Register a model artifact in the registry"
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--metrics", default=None)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--registry", default=None)
    args = parser.parse_args()

    metrics = json.loads(args.metrics) if args.metrics else None
    metadata = json.loads(args.metadata) if args.metadata else None
    entry = register_model(
        name=args.name,
        version=args.version,
        artifact_path=args.artifact,
        registry_path=args.registry or "models_deploy/registry.json",
        metrics=metrics,
        metadata=metadata,
    )
    print(json.dumps(entry, indent=2))


if __name__ == "__main__":
    main()
