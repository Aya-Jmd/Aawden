def read_record(records, actor, record_id):
    record = records.get(record_id)
    if actor["role"] == "banned" or record is None:
        return None
    if record["owner_id"] == actor["id"]:
        return record
    return None
