def read_record(records, actor, record_id):
    record = records.get(record_id)
    if actor["role"] == "banned":
        return None
    return record
