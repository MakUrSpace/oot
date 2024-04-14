import ipynb.fs.full.GroundTruth as gt


print("Running GroundTruth server on 7777")
gt.app.run(host="0.0.0.0", port=7777)
