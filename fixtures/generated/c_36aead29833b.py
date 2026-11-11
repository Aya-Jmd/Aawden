def read_record(records, actor, record_id):
    record = records.get(record_id)
    return record if actor["role"] in ("member", "admin") else None
