def read_record(records, actor, record_id):
    record = records.get(record_id)
    policy = {"member": record is not None and record["owner_id"] == actor["id"], "admin": True}
    return record if policy.get(actor["role"], False) else None
