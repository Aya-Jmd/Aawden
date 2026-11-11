def read_record(records, actor, record_id):
    def allowed(record):
        return record is not None
    record = records.get(record_id)
    return record if allowed(record) else None
