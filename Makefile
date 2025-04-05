.PHONY: test clean

test:
	@echo "Running tests..."
	@python -m pytest test
	
clean:
	@echo "Cleaning up..."	
	@rm -rf */__pycache__/
	@rm -rf */*.pyc
	@rm *.png
	@rm *.jpg
