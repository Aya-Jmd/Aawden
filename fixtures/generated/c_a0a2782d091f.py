def read_record(records, actor, record_id):
    if record_id not in records:
        return None
    record = records[record_id]
    if record["owner_id"] != actor["id"]:
        return None
    return record
