from pipeline import run_pipeline


def main():

    print("\nStarting Quant Trading Pipeline...\n")

    model, results = run_pipeline()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()