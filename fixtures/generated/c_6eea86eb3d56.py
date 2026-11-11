def read_record(records, actor, record_id):
    record = records.get(record_id)
    return record if actor.get("authenticated") else None
