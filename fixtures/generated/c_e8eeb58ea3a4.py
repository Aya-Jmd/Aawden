def read_record(records, actor, record_id):
    record = records.get(record_id)
    return record if record and (record["tenant_id"], record["owner_id"]) == (actor["tenant_id"], actor["id"]) else None
