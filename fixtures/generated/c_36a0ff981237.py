def read_record(records, actor, record_id):
    if record_id not in records:
        return None
    return records[record_id]
