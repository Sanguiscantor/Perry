PIPELINE_MODE = "multiclass"
# or
#PIPELINE_MODE = "hierarchy"


def main():

    if PIPELINE_MODE == "multiclass":
        from pipeline_multiclass import run_pipeline
    else:
        from pipeline_hierarchy import run_pipeline

    print("\nStarting Quant Trading Pipeline...\n")

    model, results = run_pipeline()

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
