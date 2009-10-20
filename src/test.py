import multiprocessing
import ann

if __name__ == "__main__":
	pool = multiprocessing.Pool(2)
	pool.map(ann.main, [["cr", 6, 1, 1.0, 10], ["cr", 12, 1, 1.0, 10]])
