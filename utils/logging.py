"""
Logging utilities for GraphHT.

This module provides a Logger class for managing logging configuration,
result reporting, and file output.
"""

import logging
import os


class Logger:
    """
    Logger class for GraphHT experiments.

    Handles logging configuration, result reporting, and file output.
    """

    def __init__(self, args):
        """
        Initialize the logger with configuration from args.

        Args:
            args: Arguments object containing experiment configuration
        """
        self.args = args
        self.logger = None
        self.log_folder_path = None
        self.log_filepath = None
        self._setup_logger()

    def _setup_logger(self):
        """Set up logging configuration and create log directories."""
        if len(self.args.hypothesis_pattern) != 1:
            raise Exception(
                f"Sorry we only support one hypothesis pattern (length == 1)."
            )

        pattern_key = str(list(self.args.hypothesis_pattern.keys())[0])

        # Set up log folder path
        self.log_folder_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "..",
            "result",
            "one_sample_log_and_results_" + pattern_key,
        )
        self.log_folder_path = os.path.normpath(self.log_folder_path)

        # Set up log file path
        self.log_filepath = os.path.join(
            self.log_folder_path,
            f"{self.args.dataset}_hypo{self.args.hypo}_{pattern_key}_{self.args.sampling_method}_{self.args.agg}_{self.args.file_num}_log.log",
        )

        # Create log directory if it doesn't exist
        if not os.path.exists(self.log_folder_path):
            os.makedirs(self.log_folder_path)

        # Configure logging
        logging.basicConfig(
            filename=self.log_filepath,
            format="%(asctime)s %(message)s",
            filemode="w",
        )

        # Create logger instance
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        # Store logger in args for backward compatibility
        self.args.logger = self.logger
        self.args.log_filepath = self.log_filepath
        self.args.log_folderPath = self.log_folder_path

    def log_global_info(self):
        """Log global experiment information."""
        self.logger.info(f"Dataset: {self.args.dataset}, Seed: {self.args.seed}")
        self.logger.info(f"Sampling Method: {self.args.sampling_method}")
        self.logger.info(f"Sampling Ratio: {self.args.sampling_percent}")
        self.logger.info(f"Hypothesis pattern: {self.args.hypothesis_pattern}")
        self.logger.info(f"Aggregation Method: {self.args.agg}")
        self.logger.info(f"=========== Start Running ===========")

    def log_hypothesis_test(self, t_stat, p_value, H0, **kwargs):
        """
        Log hypothesis testing results.

        Args:
            t_stat: Test statistic value
            p_value: P-value
            H0: Null hypothesis statement
            **kwargs: Additional arguments (twoSides or oneSide)
        """
        self.logger.info("")
        self.logger.info("[Hypothesis Testing Results]")

        if self.args.HTtype == "one-sample":
            assert len(kwargs) == 1, "Only one kwargs is allowed! eg twoSide or oneSide"
            for key, value in kwargs.items():
                if key == "twoSides":
                    self.logger.info(f"H0: {H0} == {self.args.ground_truth}.")
                    self.logger.info(f"H1: {H0} != {self.args.ground_truth}.")
                elif key == "oneSide":
                    if value == "lower":
                        self.logger.info(f"H0: {H0} = {self.args.popmean_lower}.")
                        self.logger.info(f"H1: {H0} > {self.args.popmean_lower}.")
                    elif value == "higher":
                        self.logger.info(f"H0: {H0} = {self.args.popmean_higher}.")
                        self.logger.info(f"H1: {H0} < {self.args.popmean_higher}.")
                    else:
                        raise Exception(f"Sorry, we don't support {value} for {key}.")
                else:
                    raise Exception(f"Sorry, we don't support {key}.")
        else:
            self.logger.info(f"H0: {H0}.")

        self.logger.info(f"T-statistic value: {t_stat}, P-value: {p_value}.")
        if p_value < 0.05:
            self.logger.info(
                "The test is significant, we shall reject the null hypothesis."
            )
        else:
            self.logger.info(
                "The test is NOT significant, we shall accept the null hypothesis."
            )

    def save_results(self):
        """
        Save and report hypothesis testing results.

        Outputs results to:
        1. Console (stdout)
        2. Log file (via logger)
        3. Text file (.txt format, tab-separated)

        Also handles special summary statistics for hypo == 3.
        """
        headers = [
            "Sampling time",
            "Target extraction time",
            "Total Time",
            "Accuracy",
            "node num",
            "Valid nodes/edges/paths",
            "Confidence Interval Lower",
            "Confidence Interval Upper",
            "p-value",
        ]

        # Print headers
        header_format = " | ".join([header.title().ljust(25) for header in headers])
        print(header_format)
        self.logger.info(header_format)

        list_valid = []
        txt_filepath = "_".join(self.log_filepath.split("_")[:-1]) + ".txt"
        with open(txt_filepath, "w") as file:
            file.write(
                "Sampling Time\tTarget Extraction Time\tTotal Time\tAccuracy\tSampling Ratio\tValid Nodes Edges Paths\tLower CI\tUpper CI\tp-value\n"
            )

        for index, (ratio, value) in enumerate(self.args.time_result.items()):
            (
                CI_lower,
                CI_upper,
                p_value,
                sampling_time,
                target_extraction_time,
                valid_nodes_edges_paths,
                accuracy,
            ) = value
            total_time = round(target_extraction_time + sampling_time, 2)
            list_valid.append(valid_nodes_edges_paths)

            # Print the results
            result_format = (
                f"{sampling_time:.2f}".ljust(25)
                + f"{target_extraction_time:.2f}".ljust(25)
                + f"{total_time:.2f}".ljust(25)
                + f"{accuracy:.2f}".ljust(25)
                + f"{self.args.sampling_ratio[index]}".ljust(25)
                + f"{valid_nodes_edges_paths:.2f}".ljust(25)
                + f"{CI_lower:.2f}".ljust(25)
                + f"{CI_upper:.2f}".ljust(25)
                + f"{p_value:.2f}".ljust(25)
            )
            print(result_format)
            self.logger.info(result_format)

            # Write to text file
            with open(txt_filepath, "a") as file:
                file.write(
                    f"{sampling_time:.2f}\t{target_extraction_time:.2f}\t{total_time:.2f}\t{accuracy:.2f}\t{self.args.sampling_ratio[index]}\t{valid_nodes_edges_paths:.2f}\t{CI_lower:.2f}\t{CI_upper:.2f}\t{p_value:.2f}\n"
                )

        # Handle special summary statistics for hypo == 3
        if hasattr(self.args, "hypo") and self.args.hypo == 3:
            summary_statistics_headers = [
                "User Coverage",
                "Movie Coverage",
                "Total Valid Paths",
                "Reverse Paths",
                "Self-Loops",
                "Density",
                "Diameter",
            ]

            # Print hypothesis headers
            summary_statistics_header_format = " | ".join(
                [header.title().ljust(25) for header in summary_statistics_headers]
            )
            print(summary_statistics_header_format)
            self.logger.info(summary_statistics_header_format)

            for index, (ratio, value) in enumerate(self.args.coverage.items()):
                (
                    user_coverage,
                    movie_coverage,
                    total_valid_paths,
                    total_without_reverse_paths,
                    density,
                    diameter,
                ) = value
                num_reverse_paths = round(
                    total_valid_paths - total_without_reverse_paths, 3
                )
                num_self_loops = round(
                    total_without_reverse_paths - list_valid[index], 3
                )

                # Print the hypothesis results
                hypothesis_result_format = (
                    f"{user_coverage:.3f}".ljust(25)
                    + f"{movie_coverage:.3f}".ljust(25)
                    + f"{total_valid_paths:.3f}".ljust(25)
                    + f"{num_reverse_paths:.3f}".ljust(25)
                    + f"{num_self_loops:.3f}".ljust(25)
                    + f"{density:.3f}".ljust(25)
                    + f"{diameter:.3f}".ljust(25)
                )

                print(hypothesis_result_format)
                self.logger.info(hypothesis_result_format)

        print(
            f"All hypothesis testing for ratio list {self.args.sampling_ratio} and plotting is finished!"
        )
