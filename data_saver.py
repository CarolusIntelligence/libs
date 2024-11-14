from external_libs import logging, os


class DataSaver:
    def __init__(self, data, output_path, file_format, batch_size=1000):
        self.logger = logging.getLogger(__name__)
        self.data = data
        self.output_path = output_path
        self.file_format = file_format.lower()
        self.batch_size = batch_size
        self.logger.info("DataSaver")

    def save_data_in_batches(self):
        total_rows = self.data.shape[0]
        total_batches = (total_rows // self.batch_size) + int(total_rows % self.batch_size != 0)

        for batch_num in range(total_batches):
            start = batch_num * self.batch_size
            end = min(start + self.batch_size, total_rows)
            batch_data = self.data.iloc[start:end]

            if self.file_format == "csv":
                mode = 'a' if batch_num > 0 else 'w'
                header = batch_num == 0
                batch_data.to_csv(self.output_path, mode=mode, sep=',', index=False, header=header)

            elif self.file_format == "json":
                mode = 'a'
                batch_data.to_json(self.output_path, mode=mode, orient='records', lines=True)

            elif self.file_format == "txt":
                mode = 'a' if batch_num > 0 else 'w'
                header = batch_num == 0
                batch_data.to_csv(self.output_path, mode=mode, sep='\t', index=False, header=header)

            else:
                self.logger.critical(f"file format not supported for batch writing: {self.file_format}")
                break

        self.logger.info("data writing completed")
