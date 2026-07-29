import pandas as pd
import json
from pathlib import Path
from loguru import logger
import great_expectations as gx


def validate_dataset(data_path: str, suite_path: str) -> bool:
    logger.info(f"Validating {data_path} against {suite_path}")

    df = pd.read_csv(data_path) if data_path.endswith(".csv") else pd.read_parquet(data_path)

    with open(suite_path, "r") as f:
        suite_dict = json.load(f)

    context = gx.get_context(mode="ephemeral")

    suite = context.suites.add(
        gx.ExpectationSuite(name=suite_dict["expectation_suite_name"])
    )

    expectations_map = {
        "expect_column_to_exist": gx.expectations.ExpectColumnToExist,
        "expect_column_values_to_be_between": gx.expectations.ExpectColumnValuesToBeBetween,
        "expect_column_values_to_be_in_set": gx.expectations.ExpectColumnValuesToBeInSet,
        "expect_column_values_to_not_be_null": gx.expectations.ExpectColumnValuesToNotBeNull,
        "expect_table_columns_to_match_ordered_list": gx.expectations.ExpectTableColumnsToMatchOrderedList,
        "expect_table_row_count_to_be_between": gx.expectations.ExpectTableRowCountToBeBetween,
    }

    for exp in suite_dict["expectations"]:
        exp_class = expectations_map.get(exp["expectation_type"])
        if exp_class:
            suite.add_expectation(exp_class(**exp["kwargs"]))

    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset("dataframe_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")

    validation_definition = context.validation_definitions.add(
        gx.ValidationDefinition(
            name="creditcard_validation",
            data=batch_definition,
            suite=suite,
        )
    )

    results = validation_definition.run(
        batch_parameters={"dataframe": df}
    )

    success = results.success
    total = len(results.results)
    passed = sum(1 for r in results.results if r.success)

    logger.info(f"Validation: {passed}/{total} expectations passed")

    if not success:
        failed = [r.expectation_config.expectation_type for r in results.results if not r.success]
        logger.error(f"Failed expectations: {failed}")

    return success


if __name__ == "__main__":
    import sys
    suite_path = "tests/great_expectations/creditcard_suite.json"
    result = validate_dataset("data/raw/creditcard.csv", suite_path)
    if not result:
        logger.error("Validation failed — pipeline stopped")
        sys.exit(1)
    logger.success("Validation passed")