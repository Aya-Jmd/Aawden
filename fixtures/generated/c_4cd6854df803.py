def read_record(records, actor, record_id):
    record = records.get(record_id)
    if record is None or not actor["authenticated"]:
        return None
    return record
